#!/usr/bin/env python3
"""DRL realized-Sharpe contribution analysis (counterfactual).

For every closed trade in data/trade_attribution.jsonl, look up the per-agent
signal JSONL at entry time and extract DRL's direction. Then bucket the trade
by:  DRL-aligned (sign(drl_dir) == sign(trade.direction))
      DRL-opposed (sign(drl_dir) == -sign(trade.direction))
      DRL-silent  (|drl_dir| < 0.01 OR no signal record found)

Compute per-bucket: count, win rate, mean PnL$, total PnL$, annualized Sharpe.

Constraint: all 90 closed trades have exit_time <= 2026-04-04, while DRL went
ACTIVE 2026-04-22. So this measures DRL's LATENT alpha during its SHADOW
period — what we WOULD have made if we had followed DRL signals.
"""
from __future__ import annotations
import json, math, statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRADES = REPO / "data" / "trade_attribution.jsonl"
SIG_DIR = REPO / "logs" / "attribution"

def parse_ts(s):
    if not s: raise ValueError("empty ts")
    # Strip tz so all datetimes compare as naive UTC
    for marker in ("+00:00", "Z"):
        if s.endswith(marker):
            s = s[:-len(marker)]
            break
    return datetime.fromisoformat(s)

# Load all closed trades
# [P103 2026-04-27] Bare `except: pass/continue` swallowed JSON decode
# errors and timestamp parse errors silently. Counterfactual Sharpe
# would silently drop corrupt records, biasing the alpha attribution.
# Promoted to logged + counted so operator can see corruption rate.
_parse_errors = {"trades_json": 0, "signals_json": 0, "ts_parse": 0}
trades = []
with open(TRADES) as f:
    for line in f:
        try:
            t = json.loads(line)
            if t.get("is_closed"):
                trades.append(t)
        except json.JSONDecodeError:
            _parse_errors["trades_json"] += 1

# Build signal lookup index: per (asset, day) → list of (ts, drl_dir, drl_conf)
sig_idx = defaultdict(list)
for sig_file in sorted(SIG_DIR.glob("signals_*.jsonl")):
    with open(sig_file) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                _parse_errors["signals_json"] += 1
                continue
            asset = rec.get("asset")
            ts_str = rec.get("ts")
            if not asset or not ts_str: continue
            try:
                ts = parse_ts(ts_str)
            except (ValueError, TypeError):
                _parse_errors["ts_parse"] += 1
                continue
            drl_dir = drl_conf = drl_auth = None
            for s in rec.get("signals", []):
                if s.get("agent_name") == "drl":
                    drl_dir = float(s.get("direction", 0.0))
                    drl_conf = float(s.get("confidence", 0.0))
                    drl_auth = s.get("authority", "?")
                    break
            if drl_dir is not None:
                day = ts.date().isoformat()
                sig_idx[(asset, day)].append((ts, drl_dir, drl_conf, drl_auth))

# For each trade, find DRL signal closest to entry_time
def find_drl_signal(asset, entry_ts):
    day = entry_ts.date().isoformat()
    candidates = sig_idx.get((asset, day), [])
    if not candidates:
        # Try previous day (4H bar may have logged late)
        prev = (entry_ts - timedelta(days=1)).date().isoformat()
        candidates = sig_idx.get((asset, prev), [])
    if not candidates: return None
    # Find nearest within ±2h of entry
    best = None
    best_dt = timedelta(hours=999)
    for ts, drl_dir, drl_conf, drl_auth in candidates:
        dt = abs(ts - entry_ts)
        if dt < best_dt:
            best_dt = dt
            best = (ts, drl_dir, drl_conf, drl_auth)
    if best_dt <= timedelta(hours=4):
        return best
    return None

buckets = {"aligned": [], "opposed": [], "silent": [], "no_signal": []}
authority_seen = defaultdict(int)

_skipped_no_ts = 0
for t in trades:
    if not t.get("entry_time"):
        _skipped_no_ts += 1
        continue
    try:
        entry_ts = parse_ts(t["entry_time"])
    except Exception:
        _skipped_no_ts += 1
        continue
    asset = t["asset"]
    trade_dir = int(t["direction"])  # +1 long, -1 short
    pnl = float(t["net_pnl_usd"])
    notional = float(t.get("entry_notional", 0.0))
    pnl_bps = (pnl / notional * 10000) if notional > 0 else 0.0
    sig = find_drl_signal(asset, entry_ts)
    if sig is None:
        buckets["no_signal"].append((t, pnl, pnl_bps))
        continue
    _, drl_dir, drl_conf, drl_auth = sig
    authority_seen[drl_auth] += 1
    if abs(drl_dir) < 0.05:
        buckets["silent"].append((t, pnl, pnl_bps))
    elif (drl_dir > 0 and trade_dir > 0) or (drl_dir < 0 and trade_dir < 0):
        buckets["aligned"].append((t, pnl, pnl_bps))
    else:
        buckets["opposed"].append((t, pnl, pnl_bps))

def stats(bucket_data):
    if not bucket_data: return None
    pnls = [x[1] for x in bucket_data]
    bps = [x[2] for x in bucket_data]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    mean_pnl = statistics.mean(pnls)
    total_pnl = sum(pnls)
    if n >= 2:
        std_pnl = statistics.stdev(pnls)
        # Annualize: trades span ~46 days, assume each trade ~1 day
        sharpe = (mean_pnl / std_pnl) * math.sqrt(252) if std_pnl > 0 else 0.0
    else:
        std_pnl = 0.0; sharpe = 0.0
    mean_bps = statistics.mean(bps) if bps else 0.0
    return {"n": n, "wins": wins, "win_rate_pct": wins/n*100,
            "total_pnl": total_pnl, "mean_pnl": mean_pnl, "std_pnl": std_pnl,
            "sharpe_annualized": sharpe, "mean_pnl_bps": mean_bps}

print("="*78)
print("DRL counterfactual analysis — pre-ACTIVE shadow period")
print(f"Universe: 90 closed trades, 2026-02-17 → 2026-04-04")
print(f"DRL authority during this window: {dict(authority_seen)}")
print("="*78)
print(f"{'Bucket':<14} {'N':<5} {'WinRate':<9} {'TotalPnL$':<12} {'MeanPnL$':<10} {'Sharpe':<9} {'MeanBps':<8}")
print("-"*78)
for name in ["aligned", "opposed", "silent", "no_signal"]:
    s = stats(buckets[name])
    if s is None:
        print(f"{name:<14} (empty)")
        continue
    print(f"{name:<14} {s['n']:<5} {s['win_rate_pct']:>6.1f}%  ${s['total_pnl']:+10.2f}  ${s['mean_pnl']:+8.2f}  {s['sharpe_annualized']:>+7.2f}  {s['mean_pnl_bps']:>+6.1f}")

# All-trades baseline
all_pnls = [float(t["net_pnl_usd"]) for t in trades]
all_bps = [(float(t["net_pnl_usd"])/float(t.get("entry_notional",1))*10000 if float(t.get("entry_notional",0))>0 else 0) for t in trades]
n=len(all_pnls); wins=sum(1 for p in all_pnls if p>0)
m=statistics.mean(all_pnls); sd=statistics.stdev(all_pnls)
print("-"*78)
print(f"{'ALL TRADES':<14} {n:<5} {wins/n*100:>6.1f}%  ${sum(all_pnls):+10.2f}  ${m:+8.2f}  {(m/sd)*math.sqrt(252):>+7.2f}  {statistics.mean(all_bps):>+6.1f}")

# Confidence-weighted DRL alignment
print()
print("="*78)
print("CONFIDENCE-WEIGHTED:  Did high-conviction DRL = better counterfactual?")
print("="*78)
hi_aligned = []; hi_opposed = []
lo_aligned = []; lo_opposed = []
for t in trades:
    if not t.get("entry_time"): continue
    try:
        entry_ts = parse_ts(t["entry_time"])
    except (ValueError, TypeError):
        # [P103] same parse-error tracking as above; was bare except
        _parse_errors["ts_parse"] += 1
        continue
    sig = find_drl_signal(t["asset"], entry_ts)
    if sig is None: continue
    _, drl_dir, drl_conf, _ = sig
    if abs(drl_dir) < 0.05: continue
    pnl = float(t["net_pnl_usd"])
    aligned = (drl_dir > 0 and t["direction"] > 0) or (drl_dir < 0 and t["direction"] < 0)
    if drl_conf >= 0.4:
        (hi_aligned if aligned else hi_opposed).append(pnl)
    else:
        (lo_aligned if aligned else lo_opposed).append(pnl)

for name, data in [("HIGH-CONF aligned", hi_aligned), ("HIGH-CONF opposed", hi_opposed),
                    ("LOW-CONF aligned", lo_aligned), ("LOW-CONF opposed", lo_opposed)]:
    if not data:
        print(f"  {name:<22} (empty)")
        continue
    n=len(data); wins=sum(1 for p in data if p>0)
    m=statistics.mean(data); sd=statistics.stdev(data) if n>=2 else 0
    sharpe = (m/sd)*math.sqrt(252) if sd>0 else 0
    print(f"  {name:<22} N={n:<3} winrate={wins/n*100:>5.1f}%  total=${sum(data):+9.2f}  mean=${m:+7.2f}  Sharpe={sharpe:+7.2f}")


# [P103 2026-04-27] Surface corruption visibility — was previously
# silent via bare except: pass/continue. Operator can now see if a
# significant fraction of records dropped silently.
print()
print("="*78)
_total_errs = sum(_parse_errors.values())
if _total_errs > 0:
    print(f"PARSE ERRORS (records silently dropped): {_total_errs}")
    for k, v in _parse_errors.items():
        if v > 0:
            print(f"  {k}: {v}")
else:
    print("Parse errors: 0 (clean)")
print("="*78)

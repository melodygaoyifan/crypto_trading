"""[P1R-5] Does the CURRENT configuration have edge at REAL cost? A replay.

    python -X utf8 -m analytics.backtest.current_config_replay \
        --inputs <r5_data.json> --kraken-4h <k4h.json>

PRE-REGISTERED DECISION RULE (written before the first run; the report
prints it first and it may not be edited after a result exists):
  * If, for all three assets, median gross alpha per RT <= median all-in
    cost per RT  ->  NO_EDGE_AT_CURRENT_COST.
  * If an asset's median gross alpha > median all-in cost AND the bootstrap
    95% CI lower bound of its median net per RT > 0  ->  that asset is
    EDGE_CONFIRMED_BACKTEST.
  * Otherwise UNCERTAIN, with the number of live RTs needed (from the
    bootstrap CI width) converted to weeks at the replay's RT frequency.
  * Counterfactuals (A: FLIP off; B: MIN_HOLD_BARS in {2,3,6}) can only be
    HYPOTHESIS_SUPPORTED / NOT_SUPPORTED; they never change a parameter.
  * Live expectation = replay x 0.3-0.5 (industry decay rule of thumb).

WHY NOT training/backtest_framework.BacktestEngine: its trade model enters
on every signal bar and exits after a FIXED 24 bars with overlapping trades
at 10% notional, with 0/3bps fees at 98.7% maker (the refuted model). The
sleeve is a POSITION book (hold until the seat changes, contract-quantized,
flip/resize persistence, re-entry cooldown); a per-fill cost callback would
not make that engine express it. This replay is position-based instead, and
training/ is untouched (Phase 1R iron rule 1).

WHAT IS REPLAYED (the live decision chain, imported not restated):
  regimebook book (defense.regime_book_shadow: SMA200+funding labels;
  ETH Donchian-100 per P419) -> ETF de-risk (main.etf_seat_decision on the
  defense.etf_flow_shadow z-deadband, BTC/ETH) -> skew seat (BTC,
  defense.skew_flow_signal z blend + deadband on the runtime skew bank)
  -> alpha gate (calibrated seat edge x 0.75 perf haircut vs the measured
  live threshold per asset) -> sleeve sizing (P274 equity-scaled fractions,
  floor 1 contract) -> flip / resize persistence and re-entry cooldown
  (the live profile's values).
NOT REPLAYED (stated so nobody reads the numbers as covering them): the 30s
FastRisk watchdog (its live REDUCE_50/re-add loop is a measured extra cost:
16 urgent MARKET fills since 2026-08-27), the 10% venue stop (0 fires live),
P0/trade-gate data-quality vetoes, the correlation entry block.

COSTS (per fill, expected value):
  fee     core.cde_fees schedule, tier from the replay's own rolling 30d
          traded notional; maker share m -> m*fee(maker)+(1-m)*fee(taker)
  spread  taker fills pay the Phase 0 measured median taker spread cost per
          asset (fill vs decision mid); maker fills 0
  funding CDE hourly rate from 2026-08-16 (the venue's own); before that the
          Binance daily rate as a PROXY (P218: signs can differ); a bar with
          no rate carries 0 and is COUNTED as uncovered
Gross alpha is MID-to-MID (spread is a cost line, never inside gross —
note the Phase 0 audit took gross from fill prices AND added spread, which
double-counts the spread by ~1-3bps per RT).
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ASSETS = ("BTC", "ETH", "SOL")
BAR_S = 4 * 3600
HOLD_BUCKETS = (("1", 0, 1.5), ("2", 1.5, 2.5), ("3-4", 2.5, 4.5), ("5-8", 4.5, 8.5), ("9+", 8.5, 1e9))
PERF_HAIRCUT = 0.75            # the frozen ALPHA-FEEDBACK factor (P368)

RULE_TEXT = __doc__.split("PRE-REGISTERED DECISION RULE")[1].split("WHY NOT")[0]


# ------------------------------------------------------------------ inputs
def _utc(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc)


def load_closes(kraken_4h: Optional[dict]) -> Dict[str, List[Tuple[float, float]]]:
    """(bar_close_ts, close). Binance 4H parquet, spliced with Kraken 4H bars
    after the parquet's last bar (labelled in the report)."""
    import pandas as pd
    out = {}
    for a in ASSETS:
        df = pd.read_parquet(os.path.join(REPO, "training", "training_data", "drl_training",
                                          f"{a}_4H_ohlcv.parquet"))
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        rows = [(t.timestamp() + BAR_S, float(c)) for t, c in zip(ts, df["close"])]
        last = rows[-1][0]
        for o, c in (kraken_4h or {}).get(a, []):
            if o + BAR_S > last:
                rows.append((float(o) + BAR_S, float(c)))
        out[a] = rows
    return out


def load_funding(cde_rows: list) -> Dict[str, Dict[str, Any]]:
    """Per asset: daily 8h-equivalent rates for the regimebook z, and a
    per-bar carry lookup. CDE (venue) from its first row, Binance before."""
    from training.funding_legs_lab import load_funding_daily
    out = {}
    for a in ASSETS:
        bdaily = load_funding_daily(a)
        daily = {d.date(): float(v) for d, v in zip(bdaily.index, bdaily.to_numpy())}
        cde = sorted((float(r["ts"]), float(r["funding_8h"])) for r in cde_rows
                     if r.get("asset") == a and r.get("funding_8h") is not None)
        cde_first = cde[0][0] if cde else float("inf")
        by_day: Dict[dt.date, List[float]] = {}
        for t, f8 in cde:
            by_day.setdefault(_utc(t).date(), []).append(f8)
        for d, vs in by_day.items():
            daily[d] = sum(vs) / len(vs)
        out[a] = {"daily": dict(sorted(daily.items())), "cde": cde, "cde_first": cde_first,
                  "bin_last": bdaily.index[-1].date()}
    return out


def carry_rate_bar(f: dict, t: float) -> Tuple[Optional[float], str]:
    """4H carry fraction for a bar ending at t (long PAYS a positive rate)."""
    if t >= f["cde_first"]:
        xs = [x[0] for x in f["cde"]]
        i = bisect.bisect_right(xs, t) - 1
        if i >= 0 and t - f["cde"][i][0] <= 12 * 3600:
            return f["cde"][i][1] / 2.0, "cde"
        return None, "uncovered"
    d = _utc(t).date()
    if f.get("bin_last") is not None and d <= f["bin_last"] and d in f["daily"]:
        return f["daily"][d] / 2.0, "binance_proxy"
    return None, "uncovered"


# --------------------------------------------------------------- the seats
class EtfState:
    """Daily ETF z-deadband state, stepped through the LIVE function."""
    def __init__(self, rows: list):
        from defense.etf_flow_shadow import ZSCORE_WINDOW, etf_flow_zscore_direction
        rows = sorted((r for r in rows if r.get("flow_usd") is not None),
                      key=lambda r: r["timestamp"])
        self.day_ts = [r["timestamp"] / 1000.0 for r in rows]
        flows = [float(r["flow_usd"]) for r in rows]
        self.state = []
        prev = 0.0
        for i in range(len(flows)):
            d, _z, _r = etf_flow_zscore_direction(flows[i], 0.0, flows[max(0, i - ZSCORE_WINDOW):i], prev)
            prev = d
            self.state.append(d)

    def reading(self, t: float) -> Tuple[float, bool]:
        from defense.etf_flow_shadow import MAX_AGE_DAYS
        midnight = _utc(t).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        k = bisect.bisect_left(self.day_ts, midnight) - 1
        if k < 0:
            return 0.0, False
        age = (t - (self.day_ts[k] + 86400.0)) / 86400.0
        return (self.state[k], True) if age <= MAX_AGE_DAYS else (0.0, False)


class SkewState:
    """Runtime skew bank -> live z blend + deadband (defense.skew_flow_signal)."""
    def __init__(self, rows: list):
        self.rows = sorted((r for r in rows if r.get("skew_25d") is not None), key=lambda r: r["date"])
        self.dates = [r["date"] for r in self.rows]
        self.prev = 0.0

    def step(self, t: float) -> Tuple[float, bool]:
        from defense.skew_flow_signal import (_FETCH_DAYS, _MIN_OBS, _STALE_DAYS,
                                              band_direction, zscore_trailing)
        u = _utc(t)
        cutoff = u.date().isoformat() if (u.hour * 60 + u.minute) >= 6 * 60 + 5 else \
            (u.date() - dt.timedelta(days=1)).isoformat()
        k = bisect.bisect_right(self.dates, cutoff)
        rows = self.rows[max(0, k - _FETCH_DAYS):k]
        if len(rows) < _MIN_OBS:
            return 0.0, False
        vals = [float(r["skew_25d"]) for r in rows]
        v10 = [float(r["skew_10d"]) for r in rows if r.get("skew_10d") is not None]
        z25 = zscore_trailing(vals)
        z = (z25 + zscore_trailing(v10)) / 2.0 if len(v10) >= _MIN_OBS else z25
        latest = dt.datetime.fromisoformat(rows[-1]["date"]).replace(tzinfo=dt.timezone.utc).timestamp()
        fresh = (t - latest) / 86400.0 <= _STALE_DAYS
        d = band_direction(z, self.prev)
        if fresh:
            self.prev = d
        return d, fresh


def decisions(closes, funding, etf_rows, skew_rows, start_ts, end_ts, cfg):
    """Per asset: list of (t, price, direction, decider, regime_label)."""
    import numpy as np
    from defense.regime_book_shadow import (DONCHIAN_TREND_ASSETS, FUND_Z_WINDOW, MIN_BARS,
                                            book_target, causal_funding_z, regime_label)
    from defense.trend_rule_shadow import donchian_labels
    from core.seat_alpha import REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP, skew_contra_alpha_bps
    import importlib
    # Dynamic on purpose: the seat decisions live in main.py (one
    # implementation, P172), and a STATIC import from analytics/ would pull
    # main.py's whole import graph into the CRITICAL_DIRS mypy scan (+189
    # findings measured) — the P342b scan-surface trap.
    m = importlib.import_module("main")
    etf = {a: EtfState(etf_rows.get(a) or []) for a in ("BTC", "ETH")}
    skew = {a: SkewState(skew_rows.get(a) or []) for a in skew_rows}
    out = {}
    for a in ASSETS:
        rows = closes[a]
        c = np.asarray([x[1] for x in rows], dtype=float)
        don = donchian_labels(c) if a in DONCHIAN_TREND_ASSETS else None
        days = sorted(funding[a]["daily"])
        dvals = [funding[a]["daily"][d] for d in days]
        seq = []
        for i, (t, px) in enumerate(rows):
            if t > end_ts:
                break
            # warm the stateful seats on every bar so their holds are continuous
            ed, ef = etf[a].reading(t) if a in etf else (0.0, False)
            sd, sf = skew[a].step(t) if a in skew else (0.0, False)
            if t < start_ts or i < MIN_BARS:
                continue
            reg = regime_label(list(c[i - MIN_BARS:i + 1]))
            k = bisect.bisect_left(days, _utc(t).date())
            fz = causal_funding_z(dvals[:k]) if k >= FUND_Z_WINDOW else None
            if don is not None:
                bdir, leg = (1.0, "donchian_hold") if float(don[i]) > 0 else (0.0, "donchian_flat")
            else:
                bdir, leg = book_target(a, reg, fz)
            dirn, decider, edge = float(bdir), "regimebook", REGIMEBOOK_ALPHA_BPS_PER_ROUND_TRIP.get(a, 0.0)
            es = m.etf_seat_decision(a, ed, ef, bdir, cfg["etf_derisk_assets"], cfg.get("etf_decide_assets") or [])
            if es is not None:
                decider, dirn, edge = es[0], es[1], 0.0
            ss = m.skew_seat_decision(a, sd, sf, cfg["skew_seat_assets"])
            if ss is not None:
                decider, dirn, edge = ss[0], ss[1], skew_contra_alpha_bps(a)[0]
            gate_pass = dirn == 0 or edge * PERF_HAIRCUT >= cfg["gate_thresh"][a]
            if not gate_pass:
                decider, dirn = decider + "+gate_veto", 0.0
            seq.append((t, px, dirn, decider, reg))
        out[a] = seq
    return out


# ----------------------------------------------------------------- the sim
def simulate(dec, closes, funding, sched, cfg, maker_share, spread_bps,
             equity0, flip_off=False, min_hold=0):
    """Shared-equity sleeve over the decision sequences. Returns (rts, stats)."""
    from math import floor
    cs = sched.contract_size
    fr = cfg["fractions"]
    events = sorted((t, a, px, d, dec_, reg) for a in ASSETS for (t, px, d, dec_, reg) in dec[a])
    pos = {a: 0 for a in ASSETS}
    last_px = {a: None for a in ASSETS}
    flip_streak = {a: 0 for a in ASSETS}
    resize_prop = {a: (None, 0) for a in ASSETS}
    cooldown_until = {a: -1 for a in ASSETS}
    tick_i = {a: 0 for a in ASSETS}
    held_since = {a: None for a in ASSETS}
    equity = equity0
    vol_log: List[Tuple[float, float]] = []
    open_rt: Dict[str, dict] = {}
    rts: List[dict] = []
    uncovered = covered = 0
    n_fills = 0
    fee_total = spread_total = carry_total = 0.0

    def vol30(t):
        return sum(n for (tt, n) in vol_log if t - 30 * 86400 <= tt < t)

    def fill(a, t, px, delta, reg, decider):
        nonlocal equity, n_fills, fee_total, spread_total
        if delta == 0:
            return
        n = abs(delta)
        v = vol30(t)
        fee = (maker_share * sched.fee_usd(a, n, px, True, v)
               + (1 - maker_share) * sched.fee_usd(a, n, px, False, v))
        spr = (1 - maker_share) * spread_bps[a] / 1e4 * n * cs[a] * px
        vol_log.append((t, n * cs[a] * px))
        equity -= fee + spr
        fee_total += fee; spread_total += spr; n_fills += 1
        cur = pos[a]
        rem = delta
        while rem != 0:
            p = pos[a]
            if p == 0 or (rem > 0) == (p > 0):
                part = rem
                R = open_rt.get(a)
                if R is None:
                    R = open_rt[a] = dict(asset=a, entry_ts=t, entry_notional=0.0, gross=0.0,
                                          fee=0.0, spread=0.0, carry=0.0, carry_unc=0,
                                          fills=0, regime=reg, decider=decider,
                                          direction=1 if part > 0 else -1)
                R["entry_notional"] += abs(part) * cs[a] * px
            else:
                part = -p if abs(rem) >= abs(p) else rem
            R = open_rt[a]
            share = abs(part) / n
            R["fee"] += fee * share; R["spread"] += spr * share; R["fills"] += 1
            pos[a] = p + part
            rem -= part
            if pos[a] == 0:
                R["exit_ts"] = t
                rts.append(_finish(R))
                del open_rt[a]
        _ = cur

    for (t, a, px, d, decider, reg) in events:
        # mark-to-market and carry since the last decision on this asset
        if last_px[a] is not None and pos[a] != 0:
            dpx = (px - last_px[a]) * pos[a] * cs[a]
            equity += dpx
            open_rt[a]["gross"] += dpx
            rate, src = carry_rate_bar(funding[a], t)
            if rate is None:
                uncovered += 1; open_rt[a]["carry_unc"] += 1
            else:
                covered += 1
                cc = -rate * pos[a] * cs[a] * last_px[a]
                equity += cc; carry_total += cc; open_rt[a]["carry"] += cc
        last_px[a] = px
        tick_i[a] += 1
        cur = pos[a]
        size = max(1, int(floor(fr[a] * max(equity, 0.0) / (cs[a] * px))))
        tgt = int(math.copysign(size, d)) if d != 0 else 0
        held = (tick_i[a] - held_since[a]) if (cur != 0 and held_since[a] is not None) else None
        # counterfactual B: no reduce/exit/flip before min_hold bars
        if min_hold and cur != 0 and held is not None and held < min_hold and \
                (tgt == 0 or (tgt > 0) != (cur > 0) or abs(tgt) < abs(cur)):
            continue
        if cur != 0 and tgt != 0 and (tgt > 0) != (cur > 0):          # flip
            flip_streak[a] += 1
            if flip_streak[a] < cfg["flip_persist"]:
                continue
            if flip_off:
                tgt = 0
        else:
            flip_streak[a] = 0
        if cur == 0 and tgt != 0 and tick_i[a] < cooldown_until[a]:    # re-entry cooldown
            continue
        if cur != 0 and tgt != 0 and (tgt > 0) == (cur > 0) and tgt != cur:  # resize
            pt, ps = resize_prop[a]
            ps = ps + 1 if pt == tgt else 1
            resize_prop[a] = (tgt, ps)
            if ps < cfg["resize_persist"]:
                continue
        resize_prop[a] = (None, 0)
        if tgt != cur:
            was = cur
            fill(a, t, px, tgt - cur, reg, decider)
            if tgt == 0:
                cooldown_until[a] = tick_i[a] + cfg["cooldown"] + 1
                held_since[a] = None
            elif was == 0 or (tgt > 0) != (was > 0):
                held_since[a] = tick_i[a]
    stats = dict(n_fills=n_fills, fee_usd=fee_total, spread_usd=spread_total,
                 carry_usd=carry_total, carry_bars_covered=covered,
                 carry_bars_uncovered=uncovered, end_equity=equity,
                 open_positions={a: p for a, p in pos.items() if p})
    return rts, stats


def _finish(R: dict) -> dict:
    en = R["entry_notional"]
    g = R["gross"] / en * 1e4
    fee = R["fee"] / en * 1e4
    spr = R["spread"] / en * 1e4
    fund = -R["carry"] / en * 1e4          # positive = paid
    allin = fee + spr + fund
    return dict(asset=R["asset"], entry_ts=R["entry_ts"], exit_ts=R["exit_ts"],
                direction=R["direction"], decider=R["decider"], regime=R["regime"],
                n_fills=R["fills"], hold_bars=(R["exit_ts"] - R["entry_ts"]) / BAR_S,
                entry_notional_usd=en, gross_usd=R["gross"], gross_alpha_bps=g,
                fee_bps=fee, spread_bps=spr, funding_bps=fund, all_in_cost_bps=allin,
                net_bps=g - allin, net_usd=R["gross"] - R["fee"] - R["spread"] + R["carry"],
                carry_uncovered_bars=R["carry_unc"])


# ------------------------------------------------------------- statistics
def _median(xs):
    xs = sorted(x for x in xs if x is not None and x == x)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def boot_ci_median(xs, n=4000, seed=7):
    xs = [x for x in xs if x == x]
    if len(xs) < 3:
        return None, None
    rng = random.Random(seed)
    meds = sorted(_median([rng.choice(xs) for _ in xs]) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def table(rts, key):
    g: Dict[str, List[dict]] = {}
    for r in rts:
        g.setdefault(key(r), []).append(r)
    out = {}
    for k, v in sorted(g.items()):
        out[k] = dict(n_rt=len(v),
                      median_gross_alpha_bps=_median([r["gross_alpha_bps"] for r in v]),
                      median_all_in_cost_bps=_median([r["all_in_cost_bps"] for r in v]),
                      median_net_bps=_median([r["net_bps"] for r in v]),
                      sum_net_usd=sum(r["net_usd"] for r in v),
                      win_rate_net=sum(1 for r in v if r["net_bps"] > 0) / len(v),
                      median_hold_bars=_median([r["hold_bars"] for r in v]),
                      fills_per_rt=sum(r["n_fills"] for r in v) / len(v),
                      status="OK" if len(v) >= 10 else "INSUFFICIENT")
    return out


def bucket(h):
    for name, lo, hi in HOLD_BUCKETS:
        if lo <= h < hi:
            return name
    return "9+"


def verdict(rts, span_days):
    per = {}
    asset_rts = {a: [r for r in rts if r["asset"] == a] for a in ASSETS}
    all_no_edge = True
    for a, v in asset_rts.items():
        if not v:
            per[a] = dict(verdict="NO_RT", n_rt=0)
            continue
        ga = _median([r["gross_alpha_bps"] for r in v])
        co = _median([r["all_in_cost_bps"] for r in v])
        nets = [r["net_bps"] for r in v]
        lo, hi = boot_ci_median(nets)
        if ga is not None and co is not None and ga > co:
            all_no_edge = False
        if ga is not None and co is not None and ga > co and lo is not None and lo > 0:
            vv = "EDGE_CONFIRMED_BACKTEST"
        elif ga is not None and co is not None and ga <= co:
            vv = "NO_EDGE_ASSET"
        else:
            vv = "UNCERTAIN"
        need = None
        if vv == "UNCERTAIN" and lo is not None and hi is not None and hi > lo:
            # CI half-width scales ~1/sqrt(n): RTs needed for the CI to shrink
            # below the median net (lower bound reaches 0).
            med = _median(nets)
            half = (hi - lo) / 2.0
            if med and med > 0:
                need = math.ceil(len(nets) * (half / med) ** 2)
        rt_per_week = len(v) / (span_days / 7.0)
        per[a] = dict(verdict=vv, n_rt=len(v), median_gross_alpha_bps=ga,
                      median_all_in_cost_bps=co, median_net_bps=_median(nets),
                      ci95_median_net=[lo, hi], rt_per_week=rt_per_week,
                      rts_needed=need,
                      weeks_needed=(None if need is None or rt_per_week == 0
                                    else round(max(need - len(v), 0) / rt_per_week, 1)))
    overall = "NO_EDGE_AT_CURRENT_COST" if all_no_edge else (
        "EDGE_CONFIRMED_BACKTEST(" + ",".join(a for a, x in per.items()
                                              if x.get("verdict") == "EDGE_CONFIRMED_BACKTEST") + ")"
        if any(x.get("verdict") == "EDGE_CONFIRMED_BACKTEST" for x in per.values()) else "UNCERTAIN")
    return overall, per


# ------------------------------------------------------------------- main
def live_cfg() -> dict:
    with open(os.path.join(REPO, "configs", "live_high_risk.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    return dict(fractions={a: float(c["coinbase_target_fraction_by_asset"][a]) for a in ASSETS},
                flip_persist=int(c.get("coinbase_flip_persist_ticks", 2)),
                resize_persist=int(c.get("coinbase_resize_persist_ticks", 2)),
                cooldown=int(c.get("coinbase_reentry_cooldown_ticks", 2)),
                etf_derisk_assets=list(c.get("etf_derisk_assets") or []),
                etf_decide_assets=list(c.get("etf_decide_assets") or []),
                skew_seat_assets=list(c.get("skew_seat_assets") or []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="server pull: etf/skew/cde_funding/regimes")
    ap.add_argument("--kraken-4h", default=None)
    ap.add_argument("--fills", default=os.path.join(REPO, "data", "phase0", "fills_raw.parquet"))
    ap.add_argument("--start", default="2025-10-17")
    ap.add_argument("--end", default=None)
    ap.add_argument("--equity", type=float, default=10_800.0)
    ap.add_argument("--maker-share", type=float, default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "reports", "phase1r_backtest.json"))
    a = ap.parse_args(argv)
    import pandas as pd
    from core.cde_fees import load_venue_fee_schedule
    sched = load_venue_fee_schedule(reload=True)
    if sched is None or not sched.tiers:
        print("REFUSING: tiered venue fee schedule unavailable (P1R-1)"); return 2
    with open(a.inputs, encoding="utf-8") as fh:
        inp = json.load(fh)
    k4 = None
    if a.kraken_4h:
        with open(a.kraken_4h, encoding="utf-8") as fh:
            k4 = json.load(fh)
    F = pd.read_parquet(a.fills)
    # Phase 0 measured taker spread cost per asset (fill vs decision mid)
    tk = F[F.liquidity_indicator == "TAKER"]
    spread_bps = {x: float(tk[tk.asset == x].spread_cost_bps.dropna().median()) for x in ASSETS}
    cur = F[F.iso >= "2026-08-27"]
    maker_cur = float((cur.liquidity_indicator == "MAKER").mean()) if len(cur) else 0.346
    maker = a.maker_share if a.maker_share is not None else maker_cur
    cfg = live_cfg()
    T = pd.read_json  # noqa: F841 (keeps pandas import local)
    import statistics as st
    th = {}
    gr = json.load(open(os.path.join(REPO, "data", "phase0", "gate_replay_ticks.json"), encoding="utf-8"))
    for x in ASSETS:
        v = [r["thr"] for r in gr if r["a"] == x and r.get("thr") is not None and r["thr"] < 1e6]
        th[x] = min(v) if v else float("inf")     # the LOWEST observed threshold: gate-friendliest
    cfg["gate_thresh"] = th
    closes = load_closes(k4)
    funding = load_funding(inp.get("cde_funding") or [])
    start = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc).timestamp()
    end = min(closes[x][-1][0] for x in ASSETS) if a.end is None else \
        dt.datetime.fromisoformat(a.end).replace(tzinfo=dt.timezone.utc).timestamp()
    dec = decisions(closes, funding, inp.get("etf") or {}, inp.get("skew") or {}, start, end, cfg)
    gmm: Dict[str, List[Tuple[float, str]]] = {}
    for asset, tt, reg in inp.get("regimes") or []:
        if asset in ASSETS and tt and reg:
            gmm.setdefault(asset, []).append((dt.datetime.fromisoformat(tt).timestamp(), reg))
    for x in gmm:
        gmm[x].sort()

    def gmm_at(asset, t):
        s = gmm.get(asset) or []
        i = bisect.bisect_right([y[0] for y in s], t) - 1
        return s[i][1] if i >= 0 and t - s[i][0] <= 8 * 3600 else "UNKNOWN"

    span = (end - start) / 86400.0
    runs = {}
    for name, kw in (("base", {}), ("cfA_flip_off", {"flip_off": True}),
                     ("cfB_min_hold_2", {"min_hold": 2}), ("cfB_min_hold_3", {"min_hold": 3}),
                     ("cfB_min_hold_6", {"min_hold": 6})):
        rts, stats = simulate(dec, closes, funding, sched, cfg, maker, spread_bps, a.equity, **kw)
        for r in rts:
            r["gmm_regime"] = gmm_at(r["asset"], r["entry_ts"])
        runs[name] = dict(rts=rts, stats=stats)
    sens = {}
    for m_ in (0.346, 0.697):
        rts, stats = simulate(dec, closes, funding, sched, cfg, m_, spread_bps, a.equity)
        sens[f"maker_{m_}"] = dict(verdict=verdict(rts, span), stats=stats,
                                   per_asset=table(rts, lambda r: r["asset"]))
    base = runs["base"]["rts"]
    overall, per = verdict(base, span)
    rep = dict(
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        pre_registered_rule=RULE_TEXT.strip(),
        window=dict(start=_utc(start).isoformat(), end=_utc(end).isoformat(), days=round(span, 1),
                    price_source="Binance 4H parquet to its last bar, Kraken 4H after"),
        inputs=dict(maker_share=maker, maker_share_source=("current config fills since 2026-08-27"
                                                           if a.maker_share is None else "cli"),
                    taker_spread_bps=spread_bps, gate_threshold_min_observed=th,
                    equity0=a.equity, live_cfg=cfg,
                    fee_tiers=[list(t) for t in sched.tiers],
                    fixed_per_contract={x: sched.fixed_usd_per_contract(x) for x in ASSETS}),
        verdict=overall, per_asset_verdict=per,
        decider_mix={x: _count([d[3] for d in dec[x]]) for x in ASSETS},
        tables=dict(
            per_asset=table(base, lambda r: r["asset"]),
            per_asset_hold=table(base, lambda r: f"{r['asset']} / {bucket(r['hold_bars'])}"),
            per_asset_gmm_regime=table(base, lambda r: f"{r['asset']} / {r['gmm_regime']}"),
            per_asset_book_regime=table(base, lambda r: f"{r['asset']} / {r['regime']}"),
            per_asset_decider=table(base, lambda r: f"{r['asset']} / {r['decider']}")),
        stats=runs["base"]["stats"],
        counterfactuals={},
        maker_share_sensitivity=sens,
        live_expectation_note="live ~= replay x 0.3-0.5 (industry decay rule of thumb, not a measurement)",
        round_trips=base)
    b_net = {x: sum(r["net_usd"] for r in base if r["asset"] == x) for x in ASSETS}
    for name in ("cfA_flip_off", "cfB_min_hold_2", "cfB_min_hold_3", "cfB_min_hold_6"):
        rr = runs[name]["rts"]
        c_net = {x: sum(r["net_usd"] for r in rr if r["asset"] == x) for x in ASSETS}
        d_net = {x: c_net[x] - b_net[x] for x in ASSETS}
        tot = sum(d_net.values())
        rep["counterfactuals"][name] = dict(
            n_rt={x: sum(1 for r in rr if r["asset"] == x) for x in ASSETS},
            net_usd=c_net, delta_net_usd_vs_base=d_net, delta_total_usd=tot,
            end_equity=runs[name]["stats"]["end_equity"],
            verdict="HYPOTHESIS_SUPPORTED" if tot > 0 else "NOT_SUPPORTED")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("verdict", "per_asset_verdict", "stats", "counterfactuals",
                                          "decider_mix")}, indent=1, default=str))
    return 0


def _count(xs):
    out: Dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())

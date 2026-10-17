"""[P1-6] Round-trip reconstruction from the venue's own fills — the Phase 0
cost-audit algorithm, moved into the repo so the weekly KPI report and any
Phase 0 re-run share ONE implementation (P172).

Inputs are plain data (no I/O here):
  fills      list of venue fill dicts (Coinbase `get_fills` shape:
             trade_time, product_id, side, size, price, commission,
             commission_detail_total, liquidity_indicator, order_id)
  contract_size  {asset: units per contract}
  pid_to_asset   {product_id: asset}
  decision_mid   {order_id: mid}            (data/fill_quality.jsonl)
  funding_hourly {asset: [(ts, hourly_rate), ...]}  (cde_funding_history)
  regimes        {asset: [(ts, regime_name), ...]}  (tick diagnostics)

Conventions (Phase 0 §4): FIFO on position per asset, an RT closes when the
position returns to 0 and a flip splits the fill; gross alpha / fee / spread
/ funding in bps of ENTRY notional; funding is an ESTIMATE from rates (the
venue exposes no funding-payment ledger) and is None when no rate covers the
hold; spread is fill vs decision mid, signed (positive = cost), None when no
mid exists. A missing input is never a zero (P2).
"""
from __future__ import annotations

import bisect
import datetime as dt
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):  # noqa: silent-swallow — NaN marks an unusable field
        return float("nan")


def _ts(s: str) -> float:
    return dt.datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def normalize_fills(fills: Sequence[dict], contract_size: Dict[str, float],
                    pid_to_asset: Dict[str, str],
                    decision_mid: Optional[Dict[str, float]] = None) -> List[dict]:
    out = []
    dm = decision_mid or {}
    for x in fills:
        a = pid_to_asset.get(str(x.get("product_id") or ""))
        if a is None or a not in contract_size:
            continue
        px, sz = _f(x.get("price")), _f(x.get("size"))
        cd = x.get("commission_detail_total") or {}
        notional = sz * contract_size[a] * px
        mid = dm.get(str(x.get("order_id") or ""))
        spread = None
        if mid and mid > 0:
            sgn = 1.0 if x.get("side") == "BUY" else -1.0
            spread = sgn * (px - mid) / mid * 1e4
        fixed = sum(_f(cd.get(k)) for k in ("venue_commission", "clearing_commission")
                    if not math.isnan(_f(cd.get(k))))
        out.append(dict(ts=_ts(x["trade_time"]), iso=x["trade_time"], asset=a,
                        side=x.get("side"), contracts=sz, price=px, notional=notional,
                        commission=_f(x.get("commission")),
                        client_commission=_f(cd.get("client_commission")),
                        fixed_usd=fixed, liquidity=x.get("liquidity_indicator"),
                        order_id=x.get("order_id"), spread_bps=spread))
    out.sort(key=lambda r: r["ts"])
    return out


def _lookup(series: Optional[Sequence[Tuple[float, Any]]], t: float, max_age: float):
    if not series:
        return None
    xs = [s[0] for s in series]
    i = bisect.bisect_right(xs, t) - 1
    if i < 0 or t - series[i][0] > max_age:
        return None
    return series[i][1]


def build_round_trips(fills: List[dict], contract_size: Dict[str, float],
                      funding_hourly: Optional[Dict[str, list]] = None,
                      regimes: Optional[Dict[str, list]] = None
                      ) -> Tuple[List[dict], Dict[str, dict]]:
    """Returns (closed_round_trips, open_positions_by_asset)."""
    funding_hourly = {a: sorted(v) for a, v in (funding_hourly or {}).items()}
    regimes = {a: sorted(v) for a, v in (regimes or {}).items()}
    pos: Dict[str, float] = {}
    cur: Dict[str, dict] = {}
    done: List[dict] = []
    for r in fills:
        a = r["asset"]; cs = contract_size[a]
        q = r["contracts"] * (1.0 if r["side"] == "BUY" else -1.0)
        rem = q
        while abs(rem) > 1e-12:
            p = pos.get(a, 0.0)
            if p == 0 or (rem > 0) == (p > 0):
                part, kind = rem, "entry"
            else:
                part, kind = ((-p) if abs(rem) >= abs(p) else rem), "exit"
            share = abs(part) / abs(q)
            R = cur.get(a)
            if R is None:
                R = cur[a] = dict(asset=a, entry_ts=r["ts"], entry_fills=0, exit_fills=0,
                                  entry_notional=0.0, cash=0.0, fee=0.0, spread_x=0.0,
                                  spread_missing=0, path=[], px_num=0.0, qty=0.0,
                                  fixed_dominated=0)
            nl = abs(part) * cs * r["price"]
            R["cash"] += -part * r["price"] * cs
            R["fee"] += r["commission"] * share
            if r["spread_bps"] is None:
                R["spread_missing"] += 1
            else:
                R["spread_x"] += r["spread_bps"] * nl
            if r["fixed_usd"] > (r["client_commission"] if not math.isnan(r["client_commission"]) else 0):
                R["fixed_dominated"] += 1
            if kind == "entry":
                R["entry_fills"] += 1; R["entry_notional"] += nl
                R["px_num"] += abs(part) * r["price"]; R["qty"] += abs(part)
            else:
                R["exit_fills"] += 1
            pos[a] = p + part
            R["path"].append((r["ts"], pos[a]))
            rem -= part
            if abs(pos[a]) < 1e-12:
                pos[a] = 0.0
                R["exit_ts"] = r["ts"]
                done.append(_finish(R, cs, funding_hourly.get(a), regimes.get(a)))
                del cur[a]
    open_pos = {a: dict(contracts=pos[a], entry_ts=R["entry_ts"],
                        entry_notional=R["entry_notional"]) for a, R in cur.items()}
    return done, open_pos


def _finish(R: dict, cs: float, fund: Optional[list], reg: Optional[list]) -> dict:
    en = R["entry_notional"]
    px = R["px_num"] / max(R["qty"], 1e-12)
    fund_bps: Optional[float] = None
    if fund:
        path = [(R["entry_ts"], 0.0)] + R["path"]
        cost = 0.0; total = covered = 0
        h = math.ceil(R["entry_ts"] / 3600) * 3600
        while h <= R["exit_ts"]:
            p = [pp for tt, pp in path if tt <= h][-1]
            if p != 0:
                total += 1
                rate = _lookup(fund, h, 12 * 3600)
                if rate is not None:
                    covered += 1
                    cost += rate * (1.0 if p > 0 else -1.0) * abs(p) * cs * px
            h += 3600
        if total == 0:
            fund_bps = 0.0
        elif covered == total:
            fund_bps = cost / en * 1e4
    spread = (R["spread_x"] / en) if R["spread_missing"] == 0 else None
    fee_bps = R["fee"] / en * 1e4
    allin = fee_bps + (spread or 0.0) + (fund_bps or 0.0)
    gross = R["cash"]
    hold_h = (R["exit_ts"] - R["entry_ts"]) / 3600.0
    return dict(asset=R["asset"], entry_ts=R["entry_ts"], exit_ts=R["exit_ts"],
                n_fills=R["entry_fills"] + R["exit_fills"], entry_fills=R["entry_fills"],
                exit_fills=R["exit_fills"], entry_notional_usd=en, gross_pnl_usd=gross,
                gross_alpha_bps=gross / en * 1e4, fee_usd=R["fee"], fee_bps=fee_bps,
                spread_cost_bps=spread, funding_bps=fund_bps, all_in_cost_bps=allin,
                net_bps=gross / en * 1e4 - allin, hold_hours=hold_h, hold_bars=hold_h / 4.0,
                regime_at_entry=_lookup(reg, R["entry_ts"], 8 * 3600) or "UNKNOWN",
                fixed_dominated_fills=R["fixed_dominated"])


def _median(xs: List[float]) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def kpi_table(rts: List[dict], key=lambda r: r["asset"]) -> Dict[str, dict]:
    """Per-group KPIs. n < 10 is marked INSUFFICIENT (Phase 0 rule)."""
    groups: Dict[str, List[dict]] = {}
    for r in rts:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for k, g in sorted(groups.items()):
        gross_abs = sum(abs(r["gross_pnl_usd"]) for r in g)
        out[k] = dict(
            n_rt=len(g),
            alpha_per_rt_bps=_median([r["gross_alpha_bps"] for r in g]),
            all_in_cost_per_rt_bps=_median([r["all_in_cost_bps"] for r in g]),
            net_per_rt_bps=_median([r["net_bps"] for r in g]),
            fills_per_rt=sum(r["n_fills"] for r in g) / len(g),
            median_hold_bars=_median([r["hold_bars"] for r in g]),
            fee_share_of_alpha=(sum(r["fee_usd"] for r in g) / gross_abs) if gross_abs else None,
            sum_gross_usd=sum(r["gross_pnl_usd"] for r in g),
            sum_fee_usd=sum(r["fee_usd"] for r in g),
            status="OK" if len(g) >= 10 else "INSUFFICIENT")
    return out


def render_markdown(rts: List[dict], open_pos: Dict[str, dict], since_ts: float,
                    title: str) -> str:
    cols = ["n_rt", "alpha_per_rt_bps", "all_in_cost_per_rt_bps", "net_per_rt_bps",
            "fills_per_rt", "median_hold_bars", "fee_share_of_alpha", "sum_gross_usd",
            "sum_fee_usd", "status"]

    def fmt(v):
        if v is None:
            return "—"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    def table(t):
        lines = ["| group | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for k, v in t.items():
            lines.append(f"| {k} | " + " | ".join(fmt(v[c]) for c in cols) + " |")
        return "\n".join(lines)

    win = [r for r in rts if r["exit_ts"] >= since_ts]
    since = dt.datetime.fromtimestamp(since_ts, dt.timezone.utc).date().isoformat()
    parts = [f"# {title}", "",
             f"Round trips closed since {since}: {len(win)} (all-time {len(rts)}). "
             "Funding is an ESTIMATE from rates (no venue payment ledger); spread uses the "
             "decision mid; cells with n < 10 are INSUFFICIENT.", "",
             "## Per asset (window)", table(kpi_table(win)), "",
             "## Per asset (all time)", table(kpi_table(rts)), "",
             "## Per asset x regime (all time)",
             table(kpi_table(rts, key=lambda r: f"{r['asset']} / {r['regime_at_entry']}")), "",
             "## Open positions"]
    parts += [f"- {a}: {v['contracts']:+g} ct since "
              f"{dt.datetime.fromtimestamp(v['entry_ts'], dt.timezone.utc).isoformat()}"
              for a, v in open_pos.items()] or ["- none"]
    return "\n".join(parts) + "\n"

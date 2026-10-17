"""[P1-2] Round-trip ALL-IN friction from the venue's realized costs.

    friction_rt_bps(asset, side, hold_horizon_bars) =
          rt_fee_bps(asset)        # EMA of realized fills, floored at the venue schedule
        + rt_spread_bps(asset)     # EMA of realized fill-vs-decision-mid slippage, >= 0
        + max(0, E[funding_bps])   # current hourly rate x hours held; never a credit
        + latency_rt_bps           # the gate's existing latency term, both legs

Initial values come from the Phase 0 audit (configs/cost/realized_friction_phase0.json,
written by scripts/export_venue_fee_schedule.py from reports/phase0_params.json).
Fills then update the fee and spread terms by an ASYMMETRIC EMA: up alpha=0.3,
down alpha=0.1 (costs rise fast, fall slowly).

Fail directions:
  * an unavailable live input (funding rate, fill) -> that term keeps its
    static/EMA value; the static value itself missing -> +inf (the gate then
    rejects: NO_TRADE), logged.
  * funding INCOME (a short collecting positive funding) is clipped to 0 — it
    may never lower friction to admit an entry. [OPERATOR DECISION] default.
  * the fee term is floored at 2 x the venue schedule's per-leg taker fee
    (core.cde_fees.venue_fee_bps), so the EMA can never price a round trip
    cheaper than the venue currently charges (P167).

This module is a pure cost model. It changes nothing by itself; main.py uses
it in `shadow` (log only) or `enforce` (caller-side friction injection, never
an edit of defense/constitution.py) per `realized_friction_gate_mode`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("HMATS.RealizedFriction")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_PATH = os.path.join(_REPO, "configs", "cost", "realized_friction_phase0.json")
EMA_UP = 0.3
EMA_DOWN = 0.1
HOURS_PER_BAR = 4.0


def _ema(prev: float, obs: float) -> float:
    a = EMA_UP if obs > prev else EMA_DOWN
    return prev + a * (obs - prev)


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):  # noqa: silent-swallow — None = unusable input
        return None
    return v if math.isfinite(v) else None


class RealizedFrictionModel:
    def __init__(self, static_path: Optional[str] = None,
                 state_path: Optional[str] = None):
        self.static_path = static_path or STATIC_PATH
        data_dir = os.environ.get("HMATS_DATA_DIR", "data")
        self.state_path = state_path or os.path.join(data_dir, "realized_friction_state.json")
        self.static: Dict[str, Dict[str, float]] = {}
        self.min_hold_bars: Optional[float] = None
        self.ema: Dict[str, Dict[str, float]] = {}
        self.last_ingest_ts: float = 0.0
        self._warned: set = set()
        self._load_static()
        self._load_state()

    # -- persistence -------------------------------------------------------
    def _load_static(self) -> None:
        try:
            with open(self.static_path, encoding="utf-8") as fh:
                d = json.load(fh)
            for a, v in dict(d.get("per_asset", {})).items():
                fee, spr = _finite(v.get("rt_fee_bps")), _finite(v.get("rt_spread_bps"))
                hold = _finite(v.get("median_hold_bars"))
                if fee is not None and spr is not None:
                    self.static[a.upper()] = {"rt_fee_bps": fee, "rt_spread_bps": spr,
                                              "median_hold_bars": hold or 0.0}
            self.min_hold_bars = _finite(d.get("MIN_HOLD_BARS"))
        except Exception as e:  # noqa: silent-swallow — logged; every friction() then returns +inf (fail-closed)
            logger.warning("[P1-2] static friction unavailable at %s (%s: %s) — "
                           "friction_rt_bps returns +inf (NO_TRADE)",
                           self.static_path, type(e).__name__, e)

    def _load_state(self) -> None:
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.ema = {a.upper(): {k: float(x) for k, x in v.items()}
                        for a, v in dict(d.get("ema", {})).items()}
            self.last_ingest_ts = float(d.get("last_ingest_ts", 0.0))
        except FileNotFoundError:  # noqa: silent-swallow — no state yet = cold start from the static values
            pass
        except Exception as e:  # noqa: silent-swallow — logged; EMA restarts from the static values
            logger.warning("[P1-2] friction EMA state unreadable (%s) — restarting "
                           "from Phase 0 static values", e)

    def _save_state(self) -> None:
        try:
            tmp = self.state_path + ".tmp"
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"ema": self.ema, "last_ingest_ts": self.last_ingest_ts,
                           "saved_at": time.time()}, fh)
            os.replace(tmp, self.state_path)
        except Exception as e:  # noqa: silent-swallow — logged; state is a cache of the fill ledger
            logger.warning("[P1-2] friction EMA state save failed (%s)", e)

    # -- learning ----------------------------------------------------------
    def _terms(self, a: str) -> Optional[Dict[str, float]]:
        st = self.static.get(a)
        if st is None:
            return None
        cur = self.ema.get(a)
        if cur is None:
            cur = {"rt_fee_bps": st["rt_fee_bps"], "rt_spread_bps": st["rt_spread_bps"]}
            self.ema[a] = cur
        return cur

    def update_from_fill(self, asset: str, fee_bps_leg: Optional[float],
                         slippage_bps_leg: Optional[float]) -> bool:
        a = str(asset).upper()
        cur = self._terms(a)
        if cur is None:
            return False
        f = _finite(fee_bps_leg)
        s = _finite(slippage_bps_leg)
        if f is not None and f >= 0:
            cur["rt_fee_bps"] = _ema(cur["rt_fee_bps"], 2.0 * f)
        if s is not None:
            cur["rt_spread_bps"] = _ema(cur["rt_spread_bps"], 2.0 * s)
        return f is not None or s is not None

    def ingest_fill_ledger(self, path: Optional[str] = None) -> int:
        """Fold fills newer than the last ingest from data/fill_quality.jsonl
        (venue-reported fees + decision-mid slippage, P290). Returns count."""
        p = path or os.path.join(os.environ.get("HMATS_DATA_DIR", "data"), "fill_quality.jsonl")
        n = 0
        newest = self.last_ingest_ts
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:  # noqa: silent-swallow — a torn line is skipped, the rest still counts
                        continue
                    ts = _finite(r.get("ts")) or 0.0
                    if ts <= self.last_ingest_ts or r.get("status") != "filled":
                        continue
                    px = _finite(r.get("fill_avg_price"))
                    ct = _finite(r.get("contracts"))
                    fee = _finite(r.get("fees_usd"))
                    fee_bps = None
                    if px and ct and fee is not None:
                        try:
                            from exchange.coinbase_adapter import CoinbaseAdapter
                            from exchange.symbol_mapping import SYMBOL_MAP
                            pid = SYMBOL_MAP["coinbase"]["perp"].get(str(r.get("asset")).upper())
                            cs = CoinbaseAdapter._CONTRACT_SIZE_FALLBACK.get(pid) if pid else None
                            if cs:
                                fee_bps = fee / (abs(ct) * cs * px) * 1e4
                        except Exception:  # noqa: silent-swallow — no contract size = fee term not updated for this fill
                            fee_bps = None
                    if self.update_from_fill(str(r.get("asset")), fee_bps,
                                             r.get("realized_slippage_bps")):
                        n += 1
                    newest = max(newest, ts)
        except FileNotFoundError:  # noqa: silent-swallow — no fill ledger yet: nothing to learn from
            return 0
        except Exception as e:  # noqa: silent-swallow — logged; EMA keeps its previous values
            logger.warning("[P1-2] fill-ledger ingest failed (%s)", e)
            return 0
        if newest > self.last_ingest_ts:
            self.last_ingest_ts = newest
            self._save_state()
        return n

    # -- the cost --------------------------------------------------------
    def hold_horizon_bars(self, asset: str) -> float:
        st = self.static.get(str(asset).upper()) or {}
        med = float(st.get("median_hold_bars") or 0.0)
        mh = float(self.min_hold_bars or 0.0)
        return max(mh, med, 1.0)

    def friction_rt_bps(self, asset: str, side: float, *,
                        latency_rt_bps: float,
                        funding_rate_hourly: Optional[float] = None,
                        price: Optional[float] = None,
                        hold_horizon_bars: Optional[float] = None
                        ) -> Tuple[float, Dict[str, Any]]:
        """Returns (bps, breakdown). +inf (fail-closed) when the static
        baseline for the asset is missing or anything raises."""
        a = str(asset).upper()
        try:
            cur = self._terms(a)
            if cur is None:
                if a not in self._warned:
                    self._warned.add(a)
                    logger.warning("[P1-2] %s: no Phase 0 static friction — +inf (NO_TRADE)", a)
                return math.inf, {"reason": "no_static"}
            fee = float(cur["rt_fee_bps"])
            fee_floor = None
            if price is not None:
                from core.cde_fees import venue_fee_bps
                q = venue_fee_bps(a, price, is_maker=False)
                if q is not None:
                    fee_floor = 2.0 * float(q[0])
                    fee = max(fee, fee_floor)
            spread = max(0.0, float(cur["rt_spread_bps"]))
            h = float(hold_horizon_bars) if hold_horizon_bars else self.hold_horizon_bars(a)
            r = _finite(funding_rate_hourly)
            fund_raw = None
            fund = 0.0
            if r is not None:
                sgn = 1.0 if side > 0 else (-1.0 if side < 0 else 0.0)
                fund_raw = sgn * r * HOURS_PER_BAR * h * 1e4    # longs pay a positive rate
                fund = max(0.0, fund_raw)                         # never a credit
            lat = max(0.0, float(latency_rt_bps))
            total = fee + spread + fund + lat
            return total, {"fee": fee, "fee_floor": fee_floor, "spread": spread,
                           "funding": fund, "funding_raw": fund_raw,
                           "latency": lat, "hold_bars": h}
        except Exception as e:  # noqa: silent-swallow — logged; +inf = NO_TRADE (fail-closed)
            logger.warning("[P1-2] %s friction computation failed (%s: %s) — +inf",
                           a, type(e).__name__, e)
            return math.inf, {"reason": f"error:{type(e).__name__}"}


def shadow_threshold(old_ev_threshold_bps: float, old_friction_bps: float,
                     min_alpha_bps: float, new_friction_bps: float) -> Optional[float]:
    """The gate threshold with ONLY the friction term replaced: every
    multiplier (mode N, short discount, regime) is carried by the ratio, so N
    is unchanged by construction. None when the old gate priced no friction
    (early-return paths) — there is nothing to compare."""
    of = _finite(old_friction_bps)
    oe = _finite(old_ev_threshold_bps)
    if of is None or oe is None or of <= 0:
        return None
    if not math.isfinite(new_friction_bps):
        return math.inf
    return max(float(min_alpha_bps or 0.0), oe * new_friction_bps / of)


_MODEL: Optional[RealizedFrictionModel] = None


def get_realized_friction_model() -> RealizedFrictionModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = RealizedFrictionModel()
    return _MODEL

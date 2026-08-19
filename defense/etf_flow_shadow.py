"""[P270] ETF-flow shadow strategy — BTC/ETH spot-ETF daily net flow as a
next-day direction signal, OBSERVATION-ONLY (Iron Law 7).

WHY THIS EXISTS
---------------
The 2026-08-15 research pass (external: SSRN "Price Impact of Spot Bitcoin
ETF Flows" — flows explain ~21% of daily return variance and predict
next-day returns) identified daily ETF net flow as the cheapest genuinely
NEW information basis this system is not using. It is not price-derived, so
it sits outside the measured dead zone (P258/P263: every model family on the
price/funding/flow feature basis failed).

The design-era sanity check on CoinGlass's own history (606 days through
mid-June 2026, last 60 days untouched) showed sign(flow_t) -> return_{t+1}
strongly monotone (+91bps after inflow days, -111bps after outflow days).
**That number is recorded as HYPOTHESIS, not evidence**: ETF flows finalize
with a reporting lag (T+1 for some issuers), so the API's historical day-T
row may not have been knowable at day-T close — the backtest plausibly
contains a reporting-lag leak. This harness is the honest instrument: it
records, at each live 4H tick, the direction implied by whatever the API
ACTUALLY SERVES at that moment. The forward ledger is leak-free by
construction; the ~30d P166 read is the exam. Nothing here touches orders.

MECHANICS
---------
- Source: CoinGlass v4 `/api/etf/{bitcoin,ethereum}/flow-history` (probed
  live 2026-08-15: code 0, 667/529 rows; the plan we already pay for).
- The LAST row is the in-progress day and updates intraday — using it would
  be the P253c in-progress-bar class. Only rows strictly before today's UTC
  midnight are eligible.
- Staleness bound (P265 rule: a frozen input must not keep trading into a
  forward ledger): newest completed day older than MAX_AGE_DAYS -> record
  FLAT with the reason, never a stale direction.
- Scorer contract (P236/P224): confidence = |direction| — 1.0 on a
  directional claim, 0.0 on flat, never a saturated confidence on a
  non-signal. `ts` is epoch float + `iso` (both shapes parse post-P264).
- Ledger: data/strategy_shadow/etfflow_{ASSET}.jsonl, strategy "etfflow".
  The "etfflow" prefix is registered at BOTH compute_shadow_ic default
  sites (the P192/P236 two-site rule).

Promotion bar: P166 cost-aware gate on the FORWARD ledger (IC>0 every
horizon, overlap-corrected |t|>=2, edge >= 2x round-trip) + its own P-entry
+ operator flip. SOL has no ETF; it never gets a ledger row.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

# [P310] SINGLE SOURCE for the names this module writes into a record's
# `strategy` field. Consumers (analytics/shadow_ic) must not restate
# them: P309 keyed its allowlists on LEDGER-FILE PREFIXES instead, so
# two families were silently never pooled and an archive section never
# rendered. A conformance test asserts every consumer name is one of
# these, and that every one of these is classified by a consumer.
SHADOW_STRATEGY_NAMES = frozenset({"etfflow"})


logger = logging.getLogger(__name__)

# CoinGlass v4, probed live 2026-08-15 (P218 rule: probe before design).
ETF_ENDPOINTS = {
    "BTC": "https://open-api-v4.coinglass.com/api/etf/bitcoin/flow-history",
    "ETH": "https://open-api-v4.coinglass.com/api/etf/ethereum/flow-history",
}

# Newest COMPLETED flow day older than this -> flat-with-reason. 3 days
# covers weekends (no ETF flows Sat/Sun — markets closed) without letting a
# dead feed keep asserting Thursday's direction into week two.
MAX_AGE_DAYS = 3.0

# One fetch per FETCH_TTL_SEC per asset; the signal is daily, the loop is 4H.
FETCH_TTL_SEC = 3600.0


def etf_flow_direction(flow_usd: Optional[float],
                       age_days: Optional[float],
                       max_age_days: float = MAX_AGE_DAYS,
                       ) -> Tuple[float, str]:
    """Pure decision: (direction, reason) from the newest COMPLETED day.

    Flat on: no data, stale data, or a genuinely zero flow day (markets
    closed / perfectly balanced — no information either way). Absence and
    staleness must resolve to NO CLAIM, never a held-over direction (P2).
    """
    if flow_usd is None or age_days is None:
        return 0.0, "no_data"
    if age_days > max_age_days:
        return 0.0, f"stale_{age_days:.1f}d"
    if flow_usd > 0:
        return 1.0, "inflow"
    if flow_usd < 0:
        return -1.0, "outflow"
    return 0.0, "zero_flow"


class EtfFlowShadow:
    """Self-contained (P248 pattern): fetches its own data, writes its own
    ledger, every failure is per-asset and fail-soft. A fault here can never
    touch the order path."""

    def __init__(self, data_dir: str = "data"):
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._api_key = os.environ.get("COINGLASS_API_KEY", "")
        self._cache: dict = {}          # asset -> (fetched_at, rows)
        self._warned: dict = {}         # key -> last reason (transition log)

    # ---------------- data ----------------
    def _fetch_rows(self, asset: str) -> Optional[list]:
        url = ETF_ENDPOINTS.get(asset)
        if not url or not self._api_key:
            return None
        cached = self._cache.get(asset)
        now = time.time()
        if cached and now - cached[0] < FETCH_TTL_SEC:
            return cached[1]
        try:
            req = urllib.request.Request(url, headers={
                "accept": "application/json",
                "CG-API-KEY": self._api_key,
                "coinglassSecret": self._api_key})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            rows = d.get("data") if isinstance(d, dict) else None
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"bad payload code={d.get('code')}")
            rows = sorted(rows, key=lambda x: x.get("timestamp", 0))
            self._cache[asset] = (now, rows)
            return rows
        except Exception as e:  # noqa: silent-swallow — logged via transition below; a feed outage records flat-with-reason, never kills the tick
            self._transition_log(f"fetch:{asset}",
                                 f"[ETFFLOW] {asset}: fetch failed "
                                 f"({type(e).__name__}) — serving cache/flat")
            return cached[1] if cached else None

    def latest_completed_flow(self, asset: str,
                              now_ts: Optional[float] = None,
                              ) -> Tuple[Optional[float], Optional[float],
                                         Optional[str]]:
        """(flow_usd, age_days, day_iso) of the newest COMPLETED UTC day.

        The API's last row is TODAY and updates intraday — trading on it is
        the in-progress-bar trap (P253c). Eligible rows end strictly before
        today's UTC midnight.
        """
        rows = self._fetch_rows(asset)
        if not rows:
            return None, None, None
        now = now_ts if now_ts is not None else time.time()
        midnight = (datetime.fromtimestamp(now, tz=timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .timestamp())
        completed = [r for r in rows
                     if (r.get("timestamp", 0) / 1000.0) < midnight
                     and r.get("flow_usd") is not None]
        if not completed:
            return None, None, None
        last = completed[-1]
        day_ts = last["timestamp"] / 1000.0
        # age measured from the END of the flow day (its midnight + 1d)
        age_days = (now - (day_ts + 86400.0)) / 86400.0
        day_iso = datetime.fromtimestamp(
            day_ts, tz=timezone.utc).date().isoformat()
        return float(last["flow_usd"]), max(0.0, age_days), day_iso

    # ---------------- observability ----------------
    def _transition_log(self, key: str, msg: str) -> None:
        """Log on REASON CHANGE per key, not per tick (a steady-state
        condition as a per-tick line becomes wallpaper, P202) and not
        once-per-process (week two of an outage must not be silent, P265)."""
        if self._warned.get(key) != msg:
            self._warned[key] = msg
            logger.warning(msg)

    # ---------------- ledger ----------------
    def record_tick(self, asset: str) -> Optional[dict]:
        try:
            flow, age, day_iso = self.latest_completed_flow(asset)
            direction, reason = etf_flow_direction(flow, age)
            rec = {
                "ts": time.time(),
                "iso": datetime.now(timezone.utc).isoformat(),
                "strategy": "etfflow",
                "asset": asset,
                "direction": float(direction),
                # scorer multiplies direction x confidence (P236): flat rows
                # contribute zero, never a saturated claim (P224)
                "confidence": abs(float(direction)),
                "flow_usd": flow,
                "flow_day": day_iso,
                "flow_age_days": None if age is None else round(age, 2),
                "reason": reason,
            }
            path = self._dir / f"etfflow_{asset}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
        except Exception as e:  # noqa: silent-swallow — ledger-stale-this-tick is the stated consequence; logged
            logger.warning("[ETFFLOW] %s record failed: %s — ledger stale "
                           "this tick", asset, type(e).__name__)
            return None

    def tick(self) -> list:
        """Loop-level entry point. Per-asset fail-soft; unconditional
        summary line so silence is impossible (P155)."""
        summary = []
        for asset in ETF_ENDPOINTS:
            rec = self.record_tick(asset)
            if rec:
                summary.append(f"{asset}={rec['direction']:+.0f}"
                               f"({rec['reason']},{rec['flow_day']})")
            else:
                summary.append(f"{asset}=SKIP")
        logger.info("[ETFFLOW] " + " | ".join(summary))
        return summary

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

SIGNAL [P402, upgraded from raw sign]
-------------------------------------
The ledger claim is the P400 tradeable signal, not raw sign(flow). P400's
lagged historical screen (leak-free, after honest Coinbase fees) found the
robust, fee-surviving form is a z-SCORE of the newest completed flow vs its
trailing 30 completed days, with a HOLD DEADBAND (|z|>1.0 flips; inside the
band holds the position). That deadband is what keeps turnover low (~24-33
trades/yr) so the edge clears the CDE fee floor: BTC OOS Sharpe +1.18, ETH
+1.30, both CIs excluding zero. Raw sign was the weaker, higher-turnover form
and is kept only as a secondary `raw_sign` field for A/B. The hold-state is
persisted (P154) so a restart does not reset the deadband to flat.

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

# [P402] The P400 finding: raw sign(flow) is NOT the tradeable signal. The
# robust one (OOS Sh 1.18 BTC / 1.30 ETH after honest cost, positive across
# band {0.5,1.0,1.5} x window {20,30,45}, low turnover so it survives the
# Coinbase fee) is a z-score of the newest completed flow vs its trailing
# window, with a HOLD deadband. band 1.0 / window 30 was the robust cell.
ZSCORE_WINDOW = 30      # trailing completed-flow days for the z-score
ZSCORE_BAND = 1.0       # |z| must exceed this to flip; else HOLD the position
ZSCORE_MIN_OBS = 15     # below this, warmup (no claim)


def etf_flow_direction(flow_usd: Optional[float],
                       age_days: Optional[float],
                       max_age_days: float = MAX_AGE_DAYS,
                       ) -> Tuple[float, str]:
    """Pure decision: (direction, reason) from the newest COMPLETED day.

    RAW SIGN — kept for continuity/comparison; P402 replaced it as the primary
    ledger signal with the z-score version below (P400 showed raw sign is the
    weaker, higher-turnover form). Flat on: no data, stale, or zero flow.
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


def etf_flow_zscore_direction(flow_usd: Optional[float],
                              age_days: Optional[float],
                              trailing_flows,
                              prev_direction: float,
                              max_age_days: float = MAX_AGE_DAYS,
                              band: float = ZSCORE_BAND,
                              min_obs: int = ZSCORE_MIN_OBS,
                              ) -> Tuple[float, float, str]:
    """[P402] The P400 tradeable signal: (direction, z, reason).

    z = (newest completed flow - mean(trailing)) / std(trailing), clipped +-5.
    z >  band -> +1 (strong inflow); z < -band -> -1; else HOLD prev_direction
    (the deadband — what gives it low turnover and the fee-surviving Sharpe).

    Absence/staleness/warmup resolve to NO CLAIM (0.0), never a held-over
    direction (P2). The HOLD only applies once a real z is computed. Leak-free
    by construction upstream: `flow_usd` is the newest COMPLETED day only.
    """
    if flow_usd is None or age_days is None:
        return 0.0, 0.0, "no_data"
    if age_days > max_age_days:
        return 0.0, 0.0, f"stale_{age_days:.1f}d"
    hist = [f for f in (trailing_flows or []) if f is not None]
    if len(hist) < min_obs:
        return 0.0, 0.0, "warmup"
    import statistics
    mean = statistics.fmean(hist)
    sd = statistics.pstdev(hist)
    if sd <= 0:
        return 0.0, 0.0, "zero_var"
    z = max(-5.0, min(5.0, (flow_usd - mean) / sd))
    if z > band:
        return 1.0, z, "inflow_z"
    if z < -band:
        return -1.0, z, "outflow_z"
    # inside the deadband: hold the previous position (0 if never set)
    return float(prev_direction), z, "deadband_hold"


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
        # [P402] The deadband HOLD carries a position across ticks that sit
        # inside the band — that IS the low-turnover property (P400). It must
        # survive a restart, or every deploy resets the hold to flat and the
        # turnover/Sharpe both change (P154: a limiter that re-arms on restart
        # is not a limiter). Persisted per asset.
        self._state_path = self._dir / "etfflow_state.json"
        self._last_direction: dict = self._load_state()
        # [P405] in-memory seat state for the live ETF seat: per asset,
        # (direction, fresh, ts). Set each record_tick; the seat reads it
        # one-tick-stale (a daily signal on a 4H loop — immaterial), and a
        # non-fresh or aged reading yields NO seat (fail-safe, P2).
        self._seat_state: dict = {}

    # seat freshness: a held deadband position IS a live claim; warmup/no_data/
    # stale/zero_var are NOT (absence must never seat a position, P2).
    _SEAT_FRESH_REASONS = frozenset({"inflow_z", "outflow_z", "deadband_hold"})
    _SEAT_MAX_AGE_SEC = 12 * 3600.0   # 3 ticks; a shadow that stopped updating -> no seat

    def seat_direction(self, asset: str):
        """[P405] (direction, fresh) for the live ETF seat, or None if no
        usable reading. `direction` is the z-deadband position; `fresh` means
        this tick produced a real claim (not warmup/no_data/stale) AND the
        reading is younger than _SEAT_MAX_AGE_SEC. Never fetches (reads the
        in-memory state set by the loop's record_tick) — no decision-path I/O."""
        st = self._seat_state.get(asset)
        if not st:
            return None
        direction, fresh, ts = st
        if not fresh or (time.time() - ts) > self._SEAT_MAX_AGE_SEC:
            return (float(direction), False)
        return (float(direction), True)

    def _load_state(self) -> dict:
        try:
            if self._state_path.exists():
                d = json.loads(self._state_path.read_text(encoding="utf-8"))
                return {k: float(v) for k, v in d.get("last_direction", {}).items()}
        except Exception:  # noqa: silent-swallow — corrupt state -> cold start (flat), logged; never fatal
            logger.warning("[ETFFLOW] state restore failed — cold start (flat)")
        return {}

    def _save_state(self) -> None:
        try:
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"last_direction": self._last_direction}),
                           encoding="utf-8")
            os.replace(tmp, self._state_path)
        except Exception:  # noqa: silent-swallow — a state-write failure must not kill the tick; next tick retries
            pass

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

    def trailing_completed_flows(self, asset: str, window: int,
                                 now_ts: Optional[float] = None) -> list:
        """The `window` COMPLETED flow_usd values ending BEFORE the newest
        completed day (the trailing set the z-score standardizes against —
        the newest day is the observation, not part of its own baseline).
        Same completed-day filter as latest_completed_flow (leak-free)."""
        rows = self._fetch_rows(asset)
        if not rows:
            return []
        now = now_ts if now_ts is not None else time.time()
        midnight = (datetime.fromtimestamp(now, tz=timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .timestamp())
        completed = [float(r["flow_usd"]) for r in rows
                     if (r.get("timestamp", 0) / 1000.0) < midnight
                     and r.get("flow_usd") is not None]
        # drop the newest (the observation) and take the trailing window
        return completed[:-1][-window:] if len(completed) > 1 else []

    def completed_prices(self, asset: str,
                         now_ts: Optional[float] = None) -> list:
        """[P404] Ordered completed-day `price_usd` (oldest->newest, INCLUDING
        the newest). The flow-history endpoint carries price, so SMA200 and the
        combination book are computable from the fetch we already do — no new
        feed. Same completed-day filter (leak-free)."""
        rows = self._fetch_rows(asset)
        if not rows:
            return []
        now = now_ts if now_ts is not None else time.time()
        midnight = (datetime.fromtimestamp(now, tz=timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .timestamp())
        return [float(r["price_usd"]) for r in rows
                if (r.get("timestamp", 0) / 1000.0) < midnight
                and r.get("price_usd") is not None]

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
            trailing = self.trailing_completed_flows(asset, ZSCORE_WINDOW)
            # getattr-defended: a partial restore or object.__new__ construction
            # must not break the tick path (P85). Absent -> cold start (flat).
            prev = getattr(self, "_last_direction", {}).get(asset, 0.0)
            # [P402] primary ledger claim = the P400 z-score+deadband signal
            direction, z, reason = etf_flow_zscore_direction(
                flow, age, trailing, prev)
            # HOLD-state update: only a fresh directional claim moves it. A
            # deadband tick already returns prev (so prev is unchanged); a
            # no-data/stale/warmup tick logs FLAT but leaves prev intact, so a
            # transient outage does not permanently drop the held position
            # (P400 holds through gaps; we log NO CLAIM while blind, then
            # resume). Persist so the hold survives restarts (P154).
            if reason in ("inflow_z", "outflow_z"):
                if not hasattr(self, "_last_direction"):
                    self._last_direction = {}
                self._last_direction[asset] = float(direction)
                self._save_state()
            # [P405] publish the seat reading for the live ETF seat (in-memory)
            if not hasattr(self, "_seat_state"):
                self._seat_state = {}
            self._seat_state[asset] = (
                float(direction), reason in self._SEAT_FRESH_REASONS, time.time())
            # raw sign kept as a secondary field for A/B against the old signal
            raw_dir, raw_reason = etf_flow_direction(flow, age)
            # [P404] combination shadow: SMA200 long/flat (the certified overlay,
            # here on the flow-history price as a PROXY for the live 4H sleeve)
            # + ETF-outflow de-risk. P404 measured this complementary on BTC
            # (Sh +1.49 vs SMA-alone +0.37). Observation-only; arming would gate
            # the LIVE sleeve position, not this proxy — the proxy just confirms
            # the combination holds forward and is leak-free before any flip.
            sma200 = None; sma_pos = None; combo_dir = None; combo_reason = "no_price"
            prices = self.completed_prices(asset)
            if len(prices) >= 200:
                sma200 = sum(prices[-200:]) / 200.0
                cur_px = prices[-1]
                sma_pos = 1.0 if cur_px > sma200 else 0.0
                # de-risk: long only when SMA trends AND ETF is not signaling
                # outflow (direction < 0); anything else -> flat (P404 form)
                combo_dir = 0.0 if direction < 0 else sma_pos
                combo_reason = ("etf_outflow_derisk" if direction < 0
                                else ("sma_long" if sma_pos > 0 else "sma_flat"))
            rec = {
                "ts": time.time(),
                "iso": datetime.now(timezone.utc).isoformat(),
                "strategy": "etfflow",
                "asset": asset,
                "direction": float(direction),
                # scorer multiplies direction x confidence (P236): flat rows
                # contribute zero, never a saturated claim (P224)
                "confidence": abs(float(direction)),
                "z_score": round(float(z), 3),
                "flow_usd": flow,
                "flow_day": day_iso,
                "flow_age_days": None if age is None else round(age, 2),
                "reason": reason,
                "raw_sign": float(raw_dir),
                "raw_reason": raw_reason,
                # [P404] combination-shadow fields (observation-only)
                "sma200": None if sma200 is None else round(sma200, 4),
                "sma_pos": sma_pos,
                "combo_direction": combo_dir,
                "combo_reason": combo_reason,
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
                               f"(z{rec['z_score']:+.2f},{rec['reason']},"
                               f"{rec['flow_day']})")
            else:
                summary.append(f"{asset}=SKIP")
        logger.info("[ETFFLOW] " + " | ".join(summary))
        return summary

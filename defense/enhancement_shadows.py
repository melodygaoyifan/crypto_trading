"""[P277] Enhancement shadow families — the "full enhancement" batch:
every remaining signal family with a plausible evidence path at this
system's scale, each as an OBSERVATION-ONLY forward ledger through the same
P166 exam as every prior candidate (Iron Law 7: nothing here touches
orders).

Families (2026-08-16 taxonomy audit; evidence grade in brackets):
  stablecoinflow  [M — peer-reviewed for exchange inflows, arXiv
                   2411.06327; our variable is the CLOSEST AVAILABLE PROXY:
                   daily delta of aggregate USDT market cap (mint/burn),
                   because CoinGlass's exchange-balance endpoints 500 on
                   this plan. The proxy difference is recorded on every row
                   (basis="mcap_delta_proxy") — the P166 read judges the
                   proxy, not the paper's variable.]
  oidiv           [W — OI/price divergence twins: `oidiv_confirm` (new
                   money confirms the move: OI up -> follow price) and
                   `oidiv_fade` (unwind fades the move: OI down -> fade
                   price). Disjoint conditions, both ledgered — the exam
                   decides which (if either) is real, the P219 twin
                   pattern.]
  calbasis        [W — CDE's own dated contracts vs the 20DEC30 perp-style:
                   annualized term-structure slope, signal = sign of the
                   slope's z vs its own trailing readings (change-based —
                   crypto sits in near-permanent mild contango, so a
                   level-based signal would be a constant short).]
  xsmom           [W at this breadth — cross-sectional 30d momentum across
                   the 8 book assets: long top-2, short bottom-2. First
                   expressible since the P271 breadth set; the literature
                   wants 20+ coins, which is exactly why this is an exam
                   and not a belief.]
  eventfilter     [M as a RISK filter — FOMC/CPI/Sunday-thin windows.
                   Ledger claim = the gated direction with event windows
                   zeroed. Enforce flag exists, DEFAULT OFF (P141).]

Scorer contract (P236/P224): confidence = |direction|; ts epoch + iso;
prefixes registered at BOTH compute_shadow_ic sites (P192). Absence/
staleness always resolves to flat-with-reason, never a held-over or
fabricated value (P2/P199).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

STABLE_URL = ("https://open-api-v4.coinglass.com/api/index/"
              "stableCoin-marketCap-history")
FETCH_TTL_SEC = 3600.0
XS_ASSETS = ("BTC", "ETH", "SOL", "XRP", "ADA", "LTC", "DOGE", "BNB")
XS_LOOKBACK_BARS = 180          # 30d of 4H bars
XS_TOP_K = 2

# ---------------------------------------------------------------------------
# [P277] Event calendar — STATIC, remaining 2026. FOMC decision days follow
# the Fed's published 2026 schedule; CPI release days follow the BLS
# schedule. ⚠️ OPERATOR-VERIFY before ever enabling enforcement: confirm at
# federalreserve.gov/monetarypolicy/fomccalendars.htm and bls.gov/schedule/
# news_release/cpi.htm — a wrong date here is harmless in shadow and a
# mis-timed entry block under enforcement. Windows are UTC.
FOMC_DECISION_DAYS_2026 = ("2026-09-16", "2026-10-28", "2026-12-09")
CPI_RELEASE_DAYS_2026 = ("2026-09-11", "2026-10-13", "2026-11-10",
                         "2026-12-10")
FOMC_WINDOW_UTC = (12, 22)      # block-entry hours on decision day
CPI_WINDOW_UTC = (10, 16)       # block-entry hours on release day
SUNDAY_THIN_WINDOW = True       # Sun 22:00 -> Mon 00:30 UTC


def in_event_window(now: Optional[datetime] = None) -> Optional[str]:
    """Pure: the active event-window name, or None. Day boundaries in UTC."""
    t = now or datetime.now(timezone.utc)
    day = t.date().isoformat()
    if day in FOMC_DECISION_DAYS_2026 and \
            FOMC_WINDOW_UTC[0] <= t.hour < FOMC_WINDOW_UTC[1]:
        return "fomc"
    if day in CPI_RELEASE_DAYS_2026 and \
            CPI_WINDOW_UTC[0] <= t.hour < CPI_WINDOW_UTC[1]:
        return "cpi"
    if SUNDAY_THIN_WINDOW:
        if (t.weekday() == 6 and t.hour >= 22) or \
                (t.weekday() == 0 and t.hour == 0 and t.minute <= 30):
            return "sunday_thin"
    return None


def oidiv_signals(price_chg_24h: Optional[float],
                  oi_chg_24h_pct: Optional[float],
                  thr_pct: float = 1.0):
    """Pure twins. (confirm_dir, fade_dir). Missing inputs -> both flat."""
    if price_chg_24h is None or oi_chg_24h_pct is None:
        return 0.0, 0.0
    p = 1.0 if price_chg_24h > 0 else (-1.0 if price_chg_24h < 0 else 0.0)
    if p == 0.0:
        return 0.0, 0.0
    if oi_chg_24h_pct >= thr_pct:
        return p, 0.0          # new money confirms
    if oi_chg_24h_pct <= -thr_pct:
        return 0.0, -p         # unwind fades
    return 0.0, 0.0


class EnhancementShadows:
    """Self-contained where the data is self-fetchable (stablecoin, xsmom
    closes); fed per tick where main already holds it (OI, gated direction,
    CDE quotes). Every family fail-soft and per-asset; a fault can never
    reach the order path."""

    def __init__(self, data_dir: str = "data"):
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cg_key = os.environ.get("COINGLASS_API_KEY", "")
        self._stable_cache: Optional[tuple] = None   # (fetched_at, rows)
        self._basis_hist: Dict[str, list] = {}       # asset -> slope readings
        self._warned: Dict[str, str] = {}

    # ---------------- shared ----------------
    def _write(self, strategy: str, asset: str, direction: float,
               extra: Optional[dict] = None) -> dict:
        rec = {"ts": time.time(),
               "iso": datetime.now(timezone.utc).isoformat(),
               "strategy": strategy, "asset": asset,
               "direction": float(direction),
               "confidence": abs(float(direction))}
        if extra:
            rec.update(extra)
        path = self._dir / f"{strategy}_{asset}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def _transition_log(self, key: str, msg: str) -> None:
        if self._warned.get(key) != msg:
            self._warned[key] = msg
            logger.warning(msg)

    # ---------------- stablecoinflow ----------------
    def _stable_rows(self):
        now = time.time()
        if self._stable_cache and now - self._stable_cache[0] < FETCH_TTL_SEC:
            return self._stable_cache[1]
        if not self._cg_key:
            return None
        try:
            req = urllib.request.Request(STABLE_URL, headers={
                "accept": "application/json", "CG-API-KEY": self._cg_key,
                "coinglassSecret": self._cg_key})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            data = d.get("data") or {}
            dl, tl = data.get("data_list"), data.get("time_list")
            if not dl or not tl or len(dl) != len(tl):
                raise ValueError("bad payload shape")
            rows = sorted(zip(tl, dl), key=lambda x: x[0])
            self._stable_cache = (now, rows)
            return rows
        except Exception as e:  # noqa: silent-swallow — transition-logged; a feed outage records flat-with-reason
            self._transition_log("stable_fetch",
                                 f"[ENH] stablecoin fetch failed "
                                 f"({type(e).__name__}) — serving cache/flat")
            return self._stable_cache[1] if self._stable_cache else None

    def stablecoin_direction(self, now_ts: Optional[float] = None):
        """(direction, z, reason): z of the last COMPLETED day's aggregate
        USDT-mcap delta vs its trailing 30 deltas. In-progress day excluded
        (P253c class)."""
        rows = self._stable_rows()
        if not rows:
            return 0.0, None, "no_data"
        now = now_ts if now_ts is not None else time.time()
        midnight = (datetime.fromtimestamp(now, tz=timezone.utc)
                    .replace(hour=0, minute=0, second=0, microsecond=0)
                    .timestamp())
        vals = [(t / 1000.0, sum(v.values()) if isinstance(v, dict) else 0.0)
                for t, v in rows if (t / 1000.0) < midnight]
        if len(vals) < 35:
            return 0.0, None, "insufficient_history"
        age_days = (now - (vals[-1][0] + 86400.0)) / 86400.0
        if age_days > 3.0:
            return 0.0, None, f"stale_{age_days:.1f}d"
        series = [v for _, v in vals[-32:]]
        deltas = [series[i + 1] - series[i] for i in range(len(series) - 1)]
        w = deltas[:-1]
        mu = sum(w) / len(w)
        var = sum((x - mu) ** 2 for x in w) / len(w)
        sd = var ** 0.5
        if sd <= 0:
            return 0.0, 0.0, "zero_variance"
        z = (deltas[-1] - mu) / sd
        if z > 0.5:
            return 1.0, z, "mint_inflow"
        if z < -0.5:
            return -1.0, z, "burn_outflow"
        return 0.0, z, "neutral"

    # ---------------- calbasis ----------------
    def calbasis_direction(self, asset: str, near_px: Optional[float],
                           near_days: Optional[float],
                           far_px: Optional[float],
                           far_days: Optional[float]):
        """(direction, slope, reason). Annualized slope between the nearest
        dated contract and the 2030 perp-style; signal = sign of the slope's
        z vs its own trailing readings (change-based). Young history or
        missing quotes -> flat-with-reason."""
        if not (near_px and far_px and near_days and far_days) \
                or far_days <= near_days or near_px <= 0:
            return 0.0, None, "no_quotes"
        yrs = (far_days - near_days) / 365.0
        slope = ((far_px - near_px) / near_px) / yrs
        hist = self._basis_hist.setdefault(asset, [])
        hist.append(slope)
        del hist[:-60]
        if len(hist) < 20:
            return 0.0, slope, "warmup"
        w = hist[:-1]
        mu = sum(w) / len(w)
        sd = (sum((x - mu) ** 2 for x in w) / len(w)) ** 0.5
        if sd <= 0:
            return 0.0, slope, "zero_variance"
        z = (slope - mu) / sd
        # slope RISING (steepening contango) -> crowding/carry pressure ->
        # fade; slope FALLING toward backwardation -> tightness -> follow
        if z < -0.75:
            return 1.0, slope, "tightening"
        if z > 0.75:
            return -1.0, slope, "steepening"
        return 0.0, slope, "neutral"

    # ---------------- xsmom ----------------
    def xsmom_directions(self, closes_by_asset: Dict[str, list]):
        """{asset: dir} — long top-K / short bottom-K by 30d return across
        whatever subset of the 8 has enough history THIS tick. Fewer than 6
        rankable assets -> all flat (a 3-asset 'cross-section' is noise)."""
        rets = {}
        for a, closes in (closes_by_asset or {}).items():
            if closes and len(closes) > XS_LOOKBACK_BARS and \
                    closes[-XS_LOOKBACK_BARS] > 0:
                rets[a] = closes[-1] / closes[-XS_LOOKBACK_BARS] - 1.0
        if len(rets) < 6:
            return {a: 0.0 for a in closes_by_asset or {}}
        ranked = sorted(rets, key=rets.get, reverse=True)
        out = {a: 0.0 for a in rets}
        for a in ranked[:XS_TOP_K]:
            out[a] = 1.0
        for a in ranked[-XS_TOP_K:]:
            out[a] = -1.0
        return out

    # ---------------- orchestrator ----------------
    def tick(self, oi_by_asset: Optional[dict] = None,
             gated_dir_by_asset: Optional[dict] = None,
             cde_quotes: Optional[dict] = None,
             closes_by_asset: Optional[dict] = None) -> list:
        """One loop-level call, everything fail-soft per family. Inputs are
        whatever main already holds this tick; None skips that family with
        a named reason (absence is never a fabricated flat claim — the
        family simply writes nothing that tick)."""
        summary = []
        # stablecoinflow — BTC/ETH (the paper's assets)
        try:
            d, z, reason = self.stablecoin_direction()
            for a in ("BTC", "ETH"):
                self._write("stablecoinflow", a, d,
                            {"z": None if z is None else round(z, 3),
                             "reason": reason,
                             "basis": "mcap_delta_proxy"})
            summary.append(f"stable={d:+.0f}({reason})")
        except Exception as e:  # noqa: silent-swallow — per-family fail-soft; summary shows the miss
            summary.append(f"stable=ERR({type(e).__name__})")
        # oidiv twins
        try:
            n = 0
            for a, v in (oi_by_asset or {}).items():
                c, fdir = oidiv_signals(v.get("price_chg_24h"),
                                        v.get("oi_chg_24h_pct"))
                self._write("oidiv_confirm", a, c, {"inputs": v})
                self._write("oidiv_fade", a, fdir, {"inputs": v})
                n += 1
            summary.append(f"oidiv={n}a")
        except Exception as e:  # noqa: silent-swallow — per-family fail-soft
            summary.append(f"oidiv=ERR({type(e).__name__})")
        # calbasis
        try:
            n = 0
            for a, q in (cde_quotes or {}).items():
                d, slope, reason = self.calbasis_direction(
                    a, q.get("near_px"), q.get("near_days"),
                    q.get("far_px"), q.get("far_days"))
                self._write("calbasis", a, d,
                            {"slope_ann": None if slope is None
                             else round(slope, 5), "reason": reason})
                n += 1
            summary.append(f"basis={n}a")
        except Exception as e:  # noqa: silent-swallow — per-family fail-soft
            summary.append(f"basis=ERR({type(e).__name__})")
        # xsmom
        try:
            dirs = self.xsmom_directions(closes_by_asset or {})
            for a, d in dirs.items():
                self._write("xsmom", a, d, {"universe": len(dirs)})
            summary.append(f"xsmom={len(dirs)}a")
        except Exception as e:  # noqa: silent-swallow — per-family fail-soft
            summary.append(f"xsmom=ERR({type(e).__name__})")
        # eventfilter — claim = gated direction, zeroed inside windows
        try:
            win = in_event_window()
            for a, gd in (gated_dir_by_asset or {}).items():
                d = 0.0 if win else float(gd or 0.0)
                self._write("eventfilter", a, d,
                            {"window": win or "none"})
            summary.append(f"event={win or 'none'}")
        except Exception as e:  # noqa: silent-swallow — per-family fail-soft
            summary.append(f"event=ERR({type(e).__name__})")
        logger.info("[ENH-SHADOWS] " + " | ".join(summary))
        return summary

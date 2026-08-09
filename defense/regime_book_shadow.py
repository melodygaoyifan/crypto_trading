"""[P248-GP2] Regime-book shadow harness — the P247 roster, observation-only.

Records, every 4H tick, the target position each leak-corrected regime book
would hold, to `data/strategy_shadow/regimebook_{ASSET}.jsonl` — the P219
ledger pattern, scoreable by `compute_shadow_ic`, feeding the 30d P166
forward gate. NO orders are placed or influenced (Iron Law 7).

The v1 books (everything below is computable EXACTLY at runtime from a
>=560-bar 4H close series + a 30d daily funding history):

  BTC  hold-bull / funding_short(1.0)-bear / funding_contrarian(0.5)-peace
  ETH  trend-only (hold-bull / flat elsewhere)
  SOL  hold-bull / flat-bear / flat-peace  [DEGRADED — see below]

DEGRADATION, recorded not hidden: SOL's measured book carries a
ridge_defensive bear leg (its whole edge over trend-only), but the
rt_ridge_variant_probe measured the runtime-safe feature subset at CV
-1.93% vs full-feature +5.50% — the edge lives in the denoised/external/
fv2/regime-posterior features. Shadowing a reduced variant would test a
strategy nobody measured. The bear leg therefore ships only with FULL
feature parity via the runtime DRL feature path (P214/P221/P1a parity
tests) — the explicitly scoped next build. Until then SOL's ledger rows
carry book_version="v1_degraded_no_bear_leg" so its forward IC is never
mistaken for the full book's.

Regime labels are the lab's causal a-priori definitions (SMA200 x 540-bar
momentum agreement). Funding z is CAUSAL: previous-day daily close rate,
z-scored over a trailing 30-day window (the P247-F1 convention — never a
same-day read).

Each record separates price-claim from carry so the forward review can
attribute alpha vs beta vs carry (Plan V3 rule): the claim direction is
the TARGET POSITION (what the book would hold), confidence = |target|
(P236 followup: the scorer multiplies direction x confidence — a field
the scorer multiplies by is part of the claim).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SMA_W, MOM_W = 200, 540
FUND_Z_WINDOW = 30
MIN_BARS = MOM_W + 20

BOOKS_VERSION = {
    "BTC": "v1_full",
    "ETH": "v1_full",           # trend-only IS the full measured ETH book
    "SOL": "v1_degraded_no_bear_leg",
}


def regime_label(closes) -> str:
    """Causal 3-state label from a 4H close series (last bar = now)."""
    n = len(closes)
    if n < MIN_BARS:
        return "warmup"
    sma = sum(closes[-SMA_W:]) / SMA_W
    mom = closes[-1] / closes[-MOM_W] - 1.0
    above, up = closes[-1] > sma, mom > 0
    if above and up:
        return "bull"
    if (not above) and (not up):
        return "bear"
    return "peace"


def causal_funding_z(daily_rates) -> Optional[float]:
    """z of YESTERDAY'S daily funding close vs its trailing 30d window.
    daily_rates: chronological list of daily close rates where the LAST
    entry is yesterday's completed day (the caller must never append an
    in-progress day — that is the P247-F1 leak)."""
    if daily_rates is None or len(daily_rates) < FUND_Z_WINDOW:
        return None
    w = list(daily_rates[-FUND_Z_WINDOW:])
    mu = sum(w) / len(w)
    var = sum((x - mu) ** 2 for x in w) / len(w)
    sd = var ** 0.5
    if sd <= 0:
        return 0.0
    return (w[-1] - mu) / sd


def book_target(asset: str, regime: str, funding_z: Optional[float]) -> tuple:
    """(target_position, leg_name). The p247_leakfix winners, verbatim.
    A funding cell with NO causal funding history goes FLAT with a named
    reason — absence must never read as a neutral zero signal (P2/P199)."""
    if regime == "warmup":
        return 0.0, "warmup"
    if regime == "bull":
        if asset in ("BTC", "SOL", "ETH"):
            # ETH's measured book is trend-only: hold in bull like the rest?
            # NO — ETH bull cell measured FLAT (every candidate negative);
            # trend-only's bull leg is hold. ETH book = trend-only per P247,
            # whose bull leg IS hold. BTC/SOL bull = hold.
            return (1.0, "hold") if asset != "ETH" else (1.0, "trend_hold")
    if asset == "BTC":
        if funding_z is None:
            return 0.0, "flat_no_funding_history"
        if regime == "bear":
            return (-1.0, "funding_short") if funding_z > 1.0 else (0.0, "flat")
        # peace: contrarian at 0.5
        if funding_z > 0.5:
            return -1.0, "funding_contrarian_short"
        if funding_z < -0.5:
            return 1.0, "funding_contrarian_long"
        return 0.0, "flat"
    if asset == "ETH":
        return 0.0, "trend_flat"        # trend-only: flat outside bull
    if asset == "SOL":
        return 0.0, "flat_degraded"     # bear ridge leg pending full parity
    return 0.0, "flat"


class RegimeBookShadow:
    """Per-tick recorder. Fail-soft: a broken ledger write must never
    touch the tick (but it logs — P160: writers may swallow, never
    silently)."""

    def __init__(self, data_dir: str = "data"):
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fund_hist_path = Path(data_dir) / "regimebook_funding_daily.json"
        self._fund_hist = self._load_funding_history()

    # ---------------- funding history (persisted, P154 rule) ----------
    def _load_funding_history(self):
        try:
            if self._fund_hist_path.exists():
                return json.loads(self._fund_hist_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] funding history unreadable (%s) — "
                           "cold start; funding cells flat until %d days accrue",
                           type(e).__name__, FUND_Z_WINDOW)
        return {}

    def record_daily_funding(self, asset: str, day_iso: str, rate: float):
        """Append a COMPLETED day's closing funding rate. The caller passes
        yesterday's day, never today's (P247-F1)."""
        h = self._fund_hist.setdefault(asset, {})
        if day_iso in h:
            return
        h[day_iso] = float(rate)
        # keep a bounded window
        for k in sorted(h)[:-3 * FUND_Z_WINDOW]:
            del h[k]
        try:
            tmp = self._fund_hist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._fund_hist), encoding="utf-8")
            tmp.replace(self._fund_hist_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] funding history persist failed: %s",
                           type(e).__name__)

    def _funding_series(self, asset: str):
        h = self._fund_hist.get(asset)
        if not h:
            return None
        return [h[k] for k in sorted(h)]

    # ---------------- per-tick record ---------------------------------
    def record_tick(self, asset: str, closes, price: float,
                    carry_rate_bar: Optional[float] = None):
        """Compute the book's target and append the ledger row. Returns the
        record (or None on failure) for tests/diagnostics."""
        try:
            regime = regime_label(closes)
            fz = causal_funding_z(self._funding_series(asset))
            target, leg = book_target(asset, regime, fz)
            rec = {
                "ts": time.time(),
                "iso": datetime.now(timezone.utc).isoformat(),
                "strategy": "regimebook",
                "asset": asset,
                "book_version": BOOKS_VERSION.get(asset, "unknown"),
                "regime": regime,
                "leg": leg,
                "funding_z": None if fz is None else round(fz, 4),
                "direction": float(target),
                # scorer multiplies direction x confidence (P236): |target|,
                # so flat rows contribute zero, never a saturated claim (P224)
                "confidence": abs(float(target)),
                "price": float(price),
                "carry_rate_bar": carry_rate_bar,
            }
            path = self._dir / f"regimebook_{asset}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
        except Exception as e:  # noqa: BLE001
            logger.warning("[REGIMEBOOK] %s tick record failed: %s — shadow "
                           "ledger is STALE for this tick", asset,
                           type(e).__name__)
            return None

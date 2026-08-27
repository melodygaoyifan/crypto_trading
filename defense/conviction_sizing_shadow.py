"""[WS2] Conviction-agreement SIZING forward shadow (observation-only, Iron Law 7).

The disciplined forward step of the WS2 return lever: size UP when the trend and
the LIVE P407 contrarian-skew signal AGREE (a non-euphoric uptrend), size DOWN
when skew disagrees (euphoria at a top). The 6y backtest is strong and
era-stable and the random-tier control confirms it is informative concentration,
not leverage (training/conviction_sizing_lab.py) — but it is a REAL-MONEY sizing
change, so it earns a live cutover only through forward evidence (P141).

This is a SIZING overlay, not a direction seat, so rank-IC (compute_shadow_ic)
is the wrong instrument — it writes to its OWN dir (data/conviction_shadow/) and
is judged by a PnL reader (scripts/conviction_sizing_review.py), exactly like the
lab but forward. It changes NO live position: it reads the SAME live skew signal
the P407 seat uses (parity) + its own Kraken closes, and writes only a ledger row.

Skew exists for BTC/ETH only (no SOL options), so SOL is out of scope here.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# [P172] Kraken fetch + pair map single-sourced from the sibling harnesses.
from defense.regime_book_shadow import _http_json, KRAKEN_PAIRS

logger = logging.getLogger(__name__)

ASSETS = ("BTC", "ETH")          # skew is BTC/ETH only
SMA_WIN = 200
WARMUP_BARS = SMA_WIN + 1
CAP = 2.0                        # agreement size-up (mid of the swept 1.5/2.0/2.5)
DERISK = 0.5                     # trend-long-but-skew-short -> reduce (the de-risk leg)


def _sma_last(close: np.ndarray, win: int = SMA_WIN) -> Optional[float]:
    if len(close) < win:
        return None
    return float(np.mean(close[-win:]))


def conviction_mult(trend_long: bool, skew_dir: float,
                    cap: float = CAP, derisk: float = DERISK) -> float:
    """Discrete equal-weight agree-tier multiplier (never a learned combiner)."""
    if not trend_long:
        return 0.0
    if skew_dir > 0:
        return cap          # trend AND skew agree long -> size up
    if skew_dir < 0:
        return derisk       # skew disagrees (euphoria) -> de-risk
    return 1.0              # skew neutral / not fresh -> base 1x


class ConvictionSizingShadow:
    """Per-tick recorder for the WS2 sizing overlay. Reuses the live skew
    signal (parity), fetches its own closes, fail-soft + never silent (P160)."""

    def __init__(self, skew_signal, data_dir: str = "data"):
        self._skew = skew_signal
        self._dir = Path(data_dir) / "conviction_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._warned: set = set()

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def _fetch_closes_4h(self, asset: str):
        """~720 completed 4H closes from Kraken PUBLIC OHLC; drops the
        in-progress last candle (P253c). None on any failure."""
        pair = KRAKEN_PAIRS.get(asset)
        if not pair:
            return None
        try:
            data = _http_json(
                f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval=240")
            if data.get("error"):
                raise RuntimeError(str(data["error"])[:80])
            result = data.get("result", {})
            key = next((k for k in result if k != "last"), None)
            rows = result.get(key, [])
            closes = [float(r[4]) for r in rows[:-1]]   # drop in-progress bar
            return closes if len(closes) >= WARMUP_BARS else None
        except Exception as e:  # noqa: silent-swallow — logged via _warn_once; tick skipped is the consequence
            self._warn_once(f"closes:{asset}",
                            f"[CONVSIZE] {asset}: OHLC fetch failed "
                            f"({type(e).__name__}) — tick skipped")
            return None

    def _skew_dir(self, asset: str) -> tuple:
        """(skew_dir, fresh) from the LIVE P407 seat, or (0.0, False)."""
        try:
            sd = self._skew.seat_direction(asset) if self._skew else None
            if sd is None:
                return 0.0, False
            return float(sd[0]), bool(sd[1])
        except Exception as e:  # noqa: silent-swallow — logged; a missing skew reads as neutral, never a fabricated opinion (P2)
            self._warn_once(f"skew:{asset}",
                            f"[CONVSIZE] {asset}: skew read failed "
                            f"({type(e).__name__}) — treated as neutral")
            return 0.0, False

    def record_tick(self, asset: str) -> Optional[dict]:
        closes = self._fetch_closes_4h(asset)
        if closes is None:
            return None
        close = float(closes[-1])
        sma = _sma_last(np.asarray(closes, float))
        if sma is None:
            self._warn_once(f"warmup:{asset}",
                            f"[CONVSIZE] {asset}: SMA warmup — flat row")
            trend_long, reason = False, f"warmup({len(closes)}/{WARMUP_BARS})"
        else:
            trend_long, reason = bool(close > sma), "ok"
        skew_dir, fresh = self._skew_dir(asset)
        base_pos = 1.0 if trend_long else 0.0
        conv_pos = conviction_mult(trend_long, skew_dir if fresh else 0.0)
        rec = {
            "ts": time.time(),
            "iso": pd.Timestamp.utcnow().isoformat(),
            "asset": asset,
            "close": close,
            "trend_long": trend_long,
            "skew_dir": skew_dir,
            "skew_fresh": fresh,
            "base_pos": base_pos,       # 1x trend book
            "conv_pos": conv_pos,       # conviction-sized book
            "cap": CAP,
            "derisk": DERISK,
            "reason": reason,
        }
        try:
            with open(self._dir / f"convsize_{asset}.jsonl", "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: silent-swallow — logged; a ledger write must never break a tick (Iron Law 7)
            self._warn_once(f"write:{asset}",
                            f"[CONVSIZE] {asset}: ledger write failed "
                            f"({type(e).__name__})")
        return rec

    def tick(self) -> list:
        out = []
        for a in ASSETS:
            r = self.record_tick(a)
            if r is not None:
                out.append(f"{a}=base{r['base_pos']:.0f}/conv{r['conv_pos']:.1f}"
                           + ("" if r["skew_fresh"] else "(skew stale)"))
        if out:
            logger.info("[CONVSIZE] " + " | ".join(out))
        return out

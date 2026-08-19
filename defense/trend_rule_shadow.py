"""[P289] Trend-rule challenger forward ledgers — DONCHIAN-100 + EMA-ENSEMBLE.

The two P288 dethroning candidates (they beat the SMA200 trend-only book on
ETH and SOL in-design AND pre-design at ~1/4 the turnover, and transferred to
the virgin era + all five never-fitted assets) go to their 30-day forward
exam here, the P248/P219 self-contained ledger pattern. Their certification
status is PARTIAL by the pre-committed P288 rules: both MISSED the BTC-2018
crash-dodge bar (-0.51/-0.60 vs the incumbent's -0.28), i.e. they capture
more upside and dodge crashes less — a measured risk trade-off, which is why
this is a LEDGER and not a seat. OBSERVATION-ONLY (Iron Law 7): nothing here
can touch the order path; promotion requires the P166 forward gate + an
operator risk-preference decision + its own P-entry.

CANONICAL LABELERS LIVE HERE. training/ is not shipped in the engine image
(P214), so the runtime cannot import `training/trend_rule_lab.py`; instead
the labeler math is duplicated EXACTLY and a parity test
(tests/test_p289_trend_rule_shadow.py) pins the two copies bar-for-bar — a
drift between them would forward-test a different mechanism than the lab
measured (the P164/P214 train/serve class, guarded the P192 two-file way).

DELIBERATE FIDELITY NOTES (the honest caveats, recorded up front):
  * DONCHIAN-100 is CLOSE-based, matching the lab's canonical form (its own
    docstring records the deviation from the classic high/low channel).
    Fetching highs/lows and "fixing" that here would measure a mechanism
    the P288 sweep and virgin-era probe never validated.
  * Both labelers are computed on the trailing ~720-bar Kraken window, not
    full history. EMA seeding: with adjust=False the window-start seed
    retains ~(1-2/401)^n weight — ~2.8% after 719 bars, so require
    EMAENS_WARMUP_BARS bars and emit FLAT-with-reason below it. Donchian's
    state machine is path-dependent from window start; it re-synchronizes
    with the full-history state at the first channel breakout inside the
    window (~always, over 600+ bars of crypto). The ledger judges the LIVE
    composite — that is the point (P284's note applies verbatim).
  * Trend-only expression: bull -> +1.0, everything else -> 0.0. No shorts,
    no funding legs (those are the regimebook's separate, uncertified
    question — P262). Confidence = |direction| because the scorer multiplies
    direction x confidence (P236): flat rows contribute zero, never a
    saturated claim (P224).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# [P172] fetch helper + pair map single-sourced from the sibling harness —
# a second hand-typed Kraken pair table is how DOGE's XDGUSD guess happens.
from defense.regime_book_shadow import _http_json, KRAKEN_PAIRS

logger = logging.getLogger(__name__)

# Pre-registered P288 constants — MUST stay equal to trend_rule_lab's
# (EMA_PAIRS / DON_WIN); the parity test enforces value equality too.
EMA_PAIRS = ((30, 150), (50, 200), (100, 400))
DON_WIN = 100
# EMA(400) seed weight at n bars is (1-2/401)^n: ~10.5% at 450, ~2.8% at
# 719. 450 is the floor below which the ensemble's slowest pair is mostly
# seed; normal operation (Kraken serves ~720) sits well past it.
EMAENS_WARMUP_BARS = 450
DON_WARMUP_BARS = DON_WIN + 1

# [P310] SINGLE SOURCE for the names this module writes into a record's
# `strategy` field. Consumers (analytics/shadow_ic) must not restate
# them: P309 keyed its allowlists on LEDGER-FILE PREFIXES instead, so
# two families were silently never pooled and an archive section never
# rendered. A conformance test asserts every consumer name is one of
# these, and that every one of these is classified by a consumer.
STRATEGIES = ("donchian", "emaens")
SHADOW_STRATEGY_NAMES = frozenset(STRATEGIES)
ASSETS = ("BTC", "ETH", "SOL")


# ---------------------------------------------------------------------------
# canonical labelers (exact math of trend_rule_lab.lab_donchian /
# lab_ema_ensemble — same pandas calls, same order, so labels are identical)
# ---------------------------------------------------------------------------

def donchian_labels(close: np.ndarray, win: int = DON_WIN) -> np.ndarray:
    """Close-based Donchian channel state machine: 1.0 after a close above
    the trailing `win`-bar close-high (excluding the current bar), 0.0 after
    a close below the trailing low, otherwise KEEP the prior state."""
    s = pd.Series(close)
    hi = s.rolling(win).max().shift(1).to_numpy()
    lo = s.rolling(win).min().shift(1).to_numpy()
    out = np.zeros(len(close))
    state = 0.0
    for i in range(len(close)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            state = 0.0
        elif close[i] > hi[i]:
            state = 1.0
        elif close[i] < lo[i]:
            state = 0.0
        out[i] = state
    return out


def ema_ensemble_labels(close: np.ndarray) -> np.ndarray:
    """Majority (>=2 of 3) of EMA-cross pairs (30/150), (50/200), (100/400),
    adjust=False. Bars before the slowest span are forced flat (seeded ewm
    values exist there but are not honest votes)."""
    votes = np.zeros(len(close))
    for fast, slow in EMA_PAIRS:
        ef = pd.Series(close).ewm(span=fast, adjust=False).mean().to_numpy()
        es = pd.Series(close).ewm(span=slow, adjust=False).mean().to_numpy()
        votes += (ef > es).astype(float)
    bull = votes >= 2.0
    bull[:max(s for _, s in EMA_PAIRS)] = False
    return bull.astype(float)


class TrendRuleShadow:
    """Per-tick recorder for the two challenger ledgers. Self-contained
    (fetches its own closes) and fail-soft everywhere, never silent (P160).
    """

    def __init__(self, data_dir: str = "data"):
        self._dir = Path(data_dir) / "strategy_shadow"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._warned: set = set()

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    # ---------------- self-contained data fetch ------------------------
    def fetch_closes_4h(self, asset: str):
        """~720 4H closes from Kraken's PUBLIC OHLC endpoint (the proven
        regimebook pattern). None on any failure. The IN-PROGRESS last
        candle is included by Kraken — the caller drops it (P253c)."""
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
            closes = [float(r[4]) for r in rows]
            return closes if len(closes) >= DON_WARMUP_BARS else None
        except Exception as e:  # noqa: silent-swallow — logs via _warn_once (linter cannot see method-call logging); tick skipped is the stated consequence
            self._warn_once(f"closes:{asset}",
                            f"[TRENDRULE] {asset}: OHLC fetch failed "
                            f"({type(e).__name__}) — tick skipped")
            return None

    # ---------------- per-strategy label -------------------------------
    def _label(self, strategy: str, closes) -> tuple:
        """(direction, state) for the CURRENT completed bar. Below warmup
        the honest ledger claim is flat-with-reason, never a fabricated
        direction (absence != opinion; for an observation ledger a flat row
        with the reason recorded is the truthful claim)."""
        n = len(closes)
        arr = np.asarray(closes, dtype=float)
        if strategy == "donchian":
            if n < DON_WARMUP_BARS:
                return 0.0, f"warmup({n}/{DON_WARMUP_BARS})"
            return float(donchian_labels(arr)[-1]), "ok"
        if strategy == "emaens":
            if n < EMAENS_WARMUP_BARS:
                return 0.0, f"warmup({n}/{EMAENS_WARMUP_BARS})"
            return float(ema_ensemble_labels(arr)[-1]), "ok"
        raise ValueError(f"unknown strategy {strategy!r}")

    def record_tick(self, asset: str, closes) -> Dict[str, Optional[dict]]:
        """Append one row per strategy for `asset`. Returns {strategy: rec}
        (rec None on that strategy's failure) for tests/diagnostics."""
        out: Dict[str, Optional[dict]] = {}
        for strat in STRATEGIES:
            try:
                direction, state = self._label(strat, closes)
                rec = {
                    "ts": time.time(),
                    "iso": datetime.now(timezone.utc).isoformat(),
                    "strategy": strat,
                    "asset": asset,
                    "rule_version": "v1_trend_only_p288",
                    # [P288] PARTIAL certification carried on every row so
                    # the September reader cannot mistake these for fully
                    # certified books: transfer-validated, crash-dodge miss.
                    "cert": "p288_partial",
                    "state": state,
                    "bars": len(closes),
                    "direction": direction,
                    # scorer multiplies direction x confidence (P236)
                    "confidence": abs(direction),
                    "price": float(closes[-1]) if closes else None,
                }
                path = self._dir / f"{strat}_{asset}.jsonl"
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                out[strat] = rec
            except Exception as e:  # noqa: BLE001
                logger.warning("[TRENDRULE] %s/%s record failed: %s — ledger "
                               "stale this tick", strat, asset,
                               type(e).__name__)
                out[strat] = None
        return out

    # ---------------- one-call orchestrator for main.py ----------------
    def tick(self, assets=ASSETS, closes_by_asset: Optional[dict] = None):
        """Loop-level entry point: per asset, fetch closes (or use the
        injected map — tests), drop the in-progress candle (P253c), record
        both strategies. Per-asset fail-soft; the summary line makes
        silence impossible (P155)."""
        summary = []
        for asset in assets:
            try:
                if closes_by_asset is not None:
                    closes = closes_by_asset.get(asset)
                else:
                    closes = self.fetch_closes_4h(asset)
                    if closes is not None:
                        # [P253c] Kraken's last row is the in-progress 4H
                        # candle; the lab measured completed bars only.
                        closes = closes[:-1]
                if not closes:
                    summary.append(f"{asset}=SKIP(no_closes)")
                    continue
                recs = self.record_tick(asset, closes)
                parts = [f"{s[:3]}:{r['direction']:+.0f}" if r else f"{s[:3]}:ERR"
                         for s, r in recs.items()]
                summary.append(f"{asset}={'/'.join(parts)}")
            except Exception as e:  # noqa: BLE001
                summary.append(f"{asset}=SKIP({type(e).__name__})")
        logger.info("[TRENDRULE-SHADOW] " + " | ".join(summary))

"""
================================================================================
HMATS [P293] - FRED daily macro series (replaces the missing yfinance)
================================================================================

`GlobalContextInformer` fetches DXY / US10Y / SPX / VIX / GOLD through
yfinance. **yfinance is not installed in the engine image** — a live check
returns `ModuleNotFoundError`, and the boot log has said
`MacroDataFetcher initialized (yfinance=False)` for as long as the log
survives. Every indicator therefore came from `_generate_mock_indicator`,
which is deliberately conservative (fixed values, zero change, zero z-score
-> MacroRegime.NEUTRAL) — honest, but it means the macro regime has never
been computed from a real observation.

FRED already serves the same quantities, on the key this repo already
holds, over plain HTTPS with no extra dependency:

    VIX    -> VIXCLS      (CBOE Volatility Index)
    SPX    -> SP500       (S&P 500)
    US10Y  -> DGS10       (10-Year Treasury constant maturity)
    DXY    -> DTWEXBGS    (Trade Weighted USD Index: Broad, Goods & Services)
    GOLD   -> (none)      NOT MAPPED — see below

DXY IS A LABELLED PROXY, NOT THE ICE DOLLAR INDEX. DTWEXBGS is a different
basket with a different level (~119 vs ~104) and different weights. It is
used only through its 30-day z-score, which is scale-free, so the breakout
test still means something — but anyone reading the level must know it is
not the DXY they are thinking of. Recording it beats silently substituting.

GOLD IS DELIBERATELY UNMAPPED. FRED's daily gold series (LBMA fixings) is
discontinued, and inventing a value to fill the slot is the exact defect
this module exists to remove. The caller must treat GOLD as absent.

Synchronous by design: `MacroDataFetcher._fetch_ticker_sync` already runs
inside a ThreadPoolExecutor, so a blocking stdlib request is correct there
and avoids the P253c "sync fetch on the event loop" trap.
================================================================================
"""

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("HMATS.FredMacroSeries")

# GCI ticker -> FRED series id. GOLD is absent BY DECISION (see docstring).
FRED_SERIES_MAP: Dict[str, str] = {
    "VIX": "VIXCLS",
    "SPX": "SP500",
    "US10Y": "DGS10",
    "DXY": "DTWEXBGS",
}

# Series whose FRED id is a documented PROXY rather than the named index.
PROXY_SERIES = {"DXY"}

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fred_series_for(ticker: str) -> Optional[str]:
    """FRED series id for a GCI ticker, or None if unmapped."""
    return FRED_SERIES_MAP.get(str(ticker).upper())


def fetch_daily_closes(
    ticker: str,
    api_key: Optional[str] = None,
    lookback_days: int = 60,
    timeout: float = 15.0,
) -> List[Tuple[str, float]]:
    """Fetch recent daily observations as [(YYYY-MM-DD, value)], oldest first.

    Returns an EMPTY LIST on any failure or for an unmapped ticker. An empty
    list means "no reading"; it must never be turned into a zero or into a
    placeholder level by the caller.

    FRED marks missing days (holidays, weekends) with "." — those rows are
    dropped rather than carried, so no synthetic flat day enters a z-score.
    """
    series_id = fred_series_for(ticker)
    if not series_id:
        return []

    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        logger.warning(
            "[FRED_MACRO] no FRED_API_KEY — %s has no reading (NOT a neutral value)",
            ticker,
        )
        return []

    start = (datetime.now(timezone.utc) - timedelta(days=int(lookback_days))).strftime(
        "%Y-%m-%d"
    )
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "sort_order": "asc",
    }
    url = f"{_FRED_BASE}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hmats/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning("[FRED_MACRO] %s -> HTTP %s", series_id, resp.status)
                return []
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning(
            "[FRED_MACRO] %s fetch failed: %s: %s", series_id, type(e).__name__, e
        )
        return []

    out: List[Tuple[str, float]] = []
    for row in payload.get("observations") or []:
        raw = row.get("value")
        if raw in (None, ".", ""):
            continue  # FRED's missing-day marker — drop, never carry
        try:
            val = float(raw)
        except (TypeError, ValueError):  # noqa: silent-swallow
            continue  # [P293] unparseable observation -> drop, never carry
        if val != val:  # NaN
            continue
        day = str(row.get("date") or "")
        if day:
            out.append((day, val))
    return out


def is_proxy(ticker: str) -> bool:
    """True when the FRED mapping is a labelled proxy, not the named index."""
    return str(ticker).upper() in PROXY_SERIES


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path

    if not os.environ.get("FRED_API_KEY"):
        _env = Path(__file__).resolve().parents[2] / ".env"
        if _env.exists():
            for line in _env.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("FRED_API_KEY="):
                    os.environ["FRED_API_KEY"] = line.split("=", 1)[1].strip()

    for t in ("VIX", "SPX", "US10Y", "DXY", "GOLD"):
        rows = fetch_daily_closes(t)
        tag = " [PROXY]" if is_proxy(t) else ""
        if rows:
            print(f"{t:6s}{tag}: n={len(rows):3d} last={rows[-1][0]}={rows[-1][1]}")
        else:
            print(f"{t:6s}{tag}: NO READING (unmapped or fetch failed)")

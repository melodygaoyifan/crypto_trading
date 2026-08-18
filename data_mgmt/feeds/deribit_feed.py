"""
================================================================================
HMATS [P293] - Deribit Options Feed (public, keyless)
================================================================================

Closes two measured gaps, both recorded in CLAUDE.md before this feed existed:

1. **Options put/call ratio was pinned at 1.0 forever.** P218 probed the
   CoinGlass v3 option paths (`/option/info/max-pain`, `/oi`, `/volume`) and
   found all three return HTTP 404 — the endpoints no longer exist, and the
   parent `/option/info` carries no put/call split, so PCR cannot be rebuilt
   from it at any subscription tier. The agent therefore fell back to
   `put_call_ratio=1.0`, which is indistinguishable from a perfectly balanced
   options market. Deribit is where the crypto options open interest actually
   lives, its book summary is public and keyless, and PCR is directly
   computable from it.

2. **`market_data["dvol"]` has never had a producer.** P265d found the P0
   DVOL guard (`EMERGENCY_FLAT` at dvol z>=5) reading a key nothing writes,
   and deliberately did NOT add one, because arming a never-executed
   emergency-flatten path is an activation decision (P141), not a bugfix.
   Deribit publishes DVOL — its own implied-volatility index — free. This
   feed produces it; whether it reaches `market_data` is a separate,
   default-OFF config decision made by the caller.

MEASURED COVERAGE (live probe, 2026-08-17 — do not extend by assumption):
    BTC  782 option instruments (391C/391P), DVOL live
    ETH  654 option instruments (327C/327P), DVOL live
    SOL  ZERO options, no DVOL — Deribit lists SOL as a currency but
         carries no SOL options book.

SOL therefore gets NO options reading from this feed, ever. It is left
honestly absent rather than defaulted to a neutral number: a fabricated
1.0 for SOL is exactly the failure this module exists to end (P2/P223 —
a missing value must never wear a measurement's clothes).

No API key. No account. Rate limits are generous for a 4H loop; this feed
still self-throttles on poll_interval_sec via fetch_if_stale().
================================================================================
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("HMATS.Deribit")


# Deribit lists these with an options book. SOL is deliberately absent —
# measured, not assumed (see module docstring).
SUPPORTED_OPTION_CURRENCIES = ("BTC", "ETH")


@dataclass
class DeribitOptionsMetrics:
    """Per-currency options snapshot. Every field is measured or absent."""
    currency: str
    put_call_ratio_oi: Optional[float] = None      # sum(put OI) / sum(call OI)
    put_call_ratio_volume: Optional[float] = None  # 24h volume ratio
    total_oi_calls: float = 0.0
    total_oi_puts: float = 0.0
    instrument_count: int = 0
    mean_mark_iv: Optional[float] = None
    dvol: Optional[float] = None                   # Deribit volatility index
    underlying_price: Optional[float] = None
    timestamp: Optional[datetime] = None

    def is_usable(self) -> bool:
        """True only when a real PCR was computed from a real book."""
        return (
            self.put_call_ratio_oi is not None
            and self.instrument_count > 0
            and self.total_oi_calls > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "currency": self.currency,
            "put_call_ratio_oi": self.put_call_ratio_oi,
            "put_call_ratio_volume": self.put_call_ratio_volume,
            "total_oi_calls": self.total_oi_calls,
            "total_oi_puts": self.total_oi_puts,
            "instrument_count": self.instrument_count,
            "mean_mark_iv": self.mean_mark_iv,
            "dvol": self.dvol,
            "underlying_price": self.underlying_price,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class DeribitSnapshot:
    """All currencies fetched this cycle."""
    timestamp: datetime
    metrics: Dict[str, DeribitOptionsMetrics] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def get(self, currency: str) -> Optional[DeribitOptionsMetrics]:
        return self.metrics.get(currency.upper())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "errors": list(self.errors),
        }


class DeribitFeed:
    """Public Deribit options + DVOL reader.

    Read-only. Places no orders, holds no credentials, and every failure
    degrades to "no reading" rather than to a neutral number.
    """

    BASE_URL = "https://www.deribit.com/api/v2/public"

    def __init__(
        self,
        poll_interval_sec: float = 900.0,   # 15 min; the 4H loop is far slower
        mock_mode: bool = False,
        currencies: tuple = SUPPORTED_OPTION_CURRENCIES,
    ):
        self.poll_interval_sec = poll_interval_sec
        self._mock_mode = mock_mode
        self._currencies = tuple(c.upper() for c in currencies)

        self._last_data: Optional[DeribitSnapshot] = None
        self._last_fetch_time: Optional[datetime] = None
        self._fetch_errors = 0
        self._total_fetches = 0

        logger.info(
            f"[DERIBIT] Initialized: mock={mock_mode} "
            f"currencies={','.join(self._currencies)} (public API, no key)"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_latest(self) -> Optional[DeribitSnapshot]:
        """Cached snapshot, or None if never fetched."""
        return self._last_data

    def cache_age_sec(self) -> Optional[float]:
        """Seconds since last successful fetch; None if never fetched."""
        if self._last_fetch_time is None:
            return None
        _t = self._last_fetch_time
        if _t.tzinfo is None:
            _t = _t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - _t).total_seconds()

    def get_options_metrics(self, currency: str) -> Optional[DeribitOptionsMetrics]:
        """Per-currency metrics, or None.

        None means "no reading" — for SOL that is permanent and correct.
        Callers must NOT substitute a neutral PCR of 1.0 on None; that is
        the exact defect this feed replaces.
        """
        if not self._last_data:
            return None
        return self._last_data.get(currency)

    async def fetch(self) -> Optional[DeribitSnapshot]:
        """Fetch all configured currencies. Never raises."""
        self._total_fetches += 1
        try:
            if self._mock_mode:
                return self._build_mock()
            return await self._fetch_real()
        except Exception as e:
            self._fetch_errors += 1
            logger.warning(f"[DERIBIT] Fetch failed: {type(e).__name__}: {e}")
            # Keep the previous snapshot rather than blanking it — a
            # transient network error must not read as "options vanished".
            return self._last_data

    async def fetch_if_stale(self) -> Optional[DeribitSnapshot]:
        """Fetch only when the cache is older than poll_interval_sec."""
        _age = self.cache_age_sec()
        if _age is not None and _age < self.poll_interval_sec:
            return self._last_data
        return await self.fetch()

    def get_health(self) -> Dict[str, Any]:
        return {
            "source": "deribit_public",
            "last_fetch_time": (
                self._last_fetch_time.isoformat() if self._last_fetch_time else None
            ),
            "cache_age_sec": self.cache_age_sec(),
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "currencies_with_data": sorted(
                k for k, v in (self._last_data.metrics.items() if self._last_data else [])
                if v.is_usable()
            ),
        }

    # =========================================================================
    # FETCH
    # =========================================================================

    async def _fetch_real(self) -> DeribitSnapshot:
        now = datetime.now(timezone.utc)
        snap = DeribitSnapshot(timestamp=now)

        from data_mgmt.feeds._http import create_session

        async with create_session() as session:
            for cur in self._currencies:
                try:
                    m = await self._fetch_currency(session, cur, now)
                    if m is not None:
                        snap.metrics[cur] = m
                except Exception as e:
                    # Per-currency isolation: ETH failing must not lose BTC.
                    snap.errors.append(f"{cur}:{type(e).__name__}")
                    logger.warning(f"[DERIBIT] {cur}: {type(e).__name__}: {e}")

        if not snap.metrics:
            # Nothing usable — do not overwrite a good cache with an empty
            # snapshot (the P265f "fresh-stamped zeros" defect).
            logger.warning(
                "[DERIBIT] fetch yielded no usable currency — keeping previous cache"
            )
            self._fetch_errors += 1
            return self._last_data if self._last_data else snap

        self._last_data = snap
        self._last_fetch_time = now

        _parts = []
        for cur, m in sorted(snap.metrics.items()):
            _pcr = f"{m.put_call_ratio_oi:.3f}" if m.put_call_ratio_oi is not None else "n/a"
            _dv = f"{m.dvol:.1f}" if m.dvol is not None else "n/a"
            _parts.append(f"{cur}: pcr_oi={_pcr} dvol={_dv} n={m.instrument_count}")
        logger.info("[DERIBIT] " + " | ".join(_parts))

        return snap

    async def _fetch_currency(
        self,
        session: aiohttp.ClientSession,
        currency: str,
        now: datetime,
    ) -> Optional[DeribitOptionsMetrics]:
        """Book summary + DVOL for one currency."""
        m = DeribitOptionsMetrics(currency=currency, timestamp=now)

        rows = await self._get_json(
            session,
            "get_book_summary_by_currency",
            {"currency": currency, "kind": "option"},
        )
        result = (rows or {}).get("result") or []
        if not isinstance(result, list) or not result:
            logger.warning(
                f"[DERIBIT] {currency}: no option instruments returned — "
                f"no PCR this cycle (absence is NOT a neutral 1.0)"
            )
            return None

        oi_c = oi_p = vol_c = vol_p = 0.0
        ivs: List[float] = []
        underlying: Optional[float] = None

        for row in result:
            name = str(row.get("instrument_name") or "")
            if not name:
                continue
            # Deribit instrument naming: <CUR>-<EXPIRY>-<STRIKE>-<C|P>
            side = name.rsplit("-", 1)[-1].upper()
            oi = _as_float(row.get("open_interest"))
            vol = _as_float(row.get("volume"))
            if side == "C":
                oi_c += oi
                vol_c += vol
            elif side == "P":
                oi_p += oi
                vol_p += vol
            else:
                continue
            iv = _as_float(row.get("mark_iv"), default=None)
            if iv is not None and iv > 0:
                ivs.append(iv)
            if underlying is None:
                underlying = _as_float(row.get("underlying_price"), default=None)

        m.instrument_count = len(result)
        m.total_oi_calls = oi_c
        m.total_oi_puts = oi_p
        # Division guarded: zero call OI means the ratio is undefined, not 0.
        m.put_call_ratio_oi = (oi_p / oi_c) if oi_c > 0 else None
        m.put_call_ratio_volume = (vol_p / vol_c) if vol_c > 0 else None
        m.mean_mark_iv = (sum(ivs) / len(ivs)) if ivs else None
        m.underlying_price = underlying

        # DVOL is a separate endpoint; its failure must not lose the PCR.
        try:
            m.dvol = await self._fetch_dvol(session, currency)
        except Exception as e:
            logger.warning(f"[DERIBIT] {currency} DVOL unavailable: {type(e).__name__}")

        return m

    async def _fetch_dvol(
        self, session: aiohttp.ClientSession, currency: str
    ) -> Optional[float]:
        """Latest DVOL close. Returns None rather than a placeholder."""
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - 24 * 3600 * 1000
        data = await self._get_json(
            session,
            "get_volatility_index_data",
            {
                "currency": currency,
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": 3600,
            },
        )
        rows = ((data or {}).get("result") or {}).get("data") or []
        if not rows:
            return None
        # Row format: [timestamp, open, high, low, close]
        last = rows[-1]
        if not isinstance(last, (list, tuple)) or len(last) < 5:
            return None
        return _as_float(last[4], default=None)

    async def _get_json(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.BASE_URL}/{endpoint}"
        async with session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            if resp.status != 200:
                if resp.status == 429:
                    from data_mgmt.feeds._http import parse_retry_after
                    _retry = parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        f"[DERIBIT] {endpoint} rate-limited (429), "
                        f"Retry-After={_retry}s — no reading this cycle"
                    )
                else:
                    logger.warning(f"[DERIBIT] {endpoint} -> HTTP {resp.status}")
                return None
            return await resp.json()

    # =========================================================================
    # MOCK
    # =========================================================================

    def _build_mock(self) -> DeribitSnapshot:
        """Deterministic mock. Values are LABELLED, never randomised."""
        now = datetime.now(timezone.utc)
        snap = DeribitSnapshot(timestamp=now)
        for cur in self._currencies:
            snap.metrics[cur] = DeribitOptionsMetrics(
                currency=cur,
                put_call_ratio_oi=0.60,
                put_call_ratio_volume=0.70,
                total_oi_calls=1000.0,
                total_oi_puts=600.0,
                instrument_count=2,
                mean_mark_iv=40.0,
                dvol=40.0,
                underlying_price=1000.0,
                timestamp=now,
            )
        self._last_data = snap
        self._last_fetch_time = now
        return snap


def _as_float(v: Any, default: Optional[float] = 0.0) -> Any:
    """Coerce to float; malformed values take the caller's default."""
    try:
        if v is None:
            return default
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):  # NaN/inf
            return default
        return f
    except (TypeError, ValueError):  # noqa: silent-swallow
        return default


# =============================================================================
# SINGLETON
# =============================================================================

_deribit_feed_instance: Optional[DeribitFeed] = None


def get_deribit_feed(
    poll_interval_sec: float = 900.0,
    mock_mode: bool = False,
) -> DeribitFeed:
    global _deribit_feed_instance
    if _deribit_feed_instance is None:
        _deribit_feed_instance = DeribitFeed(
            poll_interval_sec=poll_interval_sec, mock_mode=mock_mode
        )
    return _deribit_feed_instance


def reset_deribit_feed():
    global _deribit_feed_instance
    _deribit_feed_instance = None


if __name__ == "__main__":
    async def _main():
        logging.basicConfig(level=logging.INFO)
        feed = DeribitFeed()
        snap = await feed.fetch()
        if snap:
            import json
            print(json.dumps(snap.to_dict(), indent=2, default=str))
        print("health:", feed.get_health())

    asyncio.run(_main())

"""
================================================================================
HMATS [P293] - Exchange Netflow Feed (CoinGlass v4, existing key)
================================================================================

Closes the measured gap: `exchange_flow` has been STRUCTURALLY ZERO for the
whole life of the system.

The only producer is `onchain_feed._parse_raw_data`, which computes
    net_exchange_flow = sum(f.value_usd ... for f in exchange_flows)
over `raw["exchange_flows"]` — a list no configured source has ever
populated (Blockchair publishes no exchange labels; the CryptoCompare path
carries none either). Summing an empty list yields 0.0, and the consumer
needs |flow| > $1M to emit a direction, so `flow_direction` was pinned at
0.0 forever — on an agent carrying ADVISE weight 0.20 in fusion.

A live probe (2026-08-17) found the data on the key we ALREADY pay for:
    GET https://open-api-v4.coinglass.com/api/exchange/balance/list?symbol=BTC
returns per-exchange `total_balance` plus `balance_change_1d/7d/30d`.

MEASURED COVERAGE (probe, do not extend by assumption):
    BTC  21 exchanges, total 2,510,258 BTC, net_1d +4,046 BTC
    ETH  20 exchanges, total 12,065,649 ETH, net_1d -13,595 ETH
    SOL  ZERO exchanges — NOT COVERED

SOL therefore has no netflow reading and must keep emitting nothing. A
fabricated 0.0 for SOL would be indistinguishable from a measured "flat
flows" reading, which is the defect this module exists to end.

--------------------------------------------------------------------------
SIGN CONVENTION — read this before wiring anything to it
--------------------------------------------------------------------------
This module reports `netflow_coins` / `netflow_usd` where

        POSITIVE = NET INFLOW TO EXCHANGES  (coins moving onto venues,
                   classically read as SELL pressure -> BEARISH)
        NEGATIVE = NET OUTFLOW FROM EXCHANGES (-> BULLISH)

matching the existing PRODUCER convention in onchain_feed (`inflow` adds).

**The legacy consumer in main.py uses the OPPOSITE convention.** It reads
    flow_direction = +1.0 if exchange_flow > 1e6 else -1.0 if < -1e6
under a comment reading "exchange outflow = bullish, inflow = bearish",
i.e. it expects POSITIVE to mean OUTFLOW. Producer and consumer disagree
by a sign, and this has never surfaced only because the producer has
always returned exactly 0.0.

Consequently this feed does NOT write the legacy `exchange_flow` key. It
exposes its own unambiguous fields, and any bridge into the legacy pair
must negate explicitly and be pinned by a test. Arming a sign-inverted
flow signal would be worse than the zero it replaces.
================================================================================
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("HMATS.ExchangeNetflow")


# Measured as covered by the endpoint. SOL is absent BY MEASUREMENT.
SUPPORTED_SYMBOLS = ("BTC", "ETH")


@dataclass
class ExchangeNetflowMetrics:
    """Per-symbol exchange balance snapshot.

    netflow_* are POSITIVE FOR INFLOW (bearish). See module docstring.
    """
    symbol: str
    total_balance_coins: float = 0.0
    netflow_coins_1d: float = 0.0
    netflow_coins_7d: float = 0.0
    exchange_count: int = 0
    timestamp: Optional[datetime] = None
    # [P294] Provenance travels WITH the reading. Mock mode emits netflow
    # 0.0 with exchange_count=1, so `is_usable()` is True and the value is
    # byte-identical to a measured "flows were flat" — the exact
    # missing-vs-measured collapse this module's own docstring exists to
    # end, reintroduced by its own fallback. A consumer that cares can now
    # tell; one that does not is unaffected, since 0.0 is also the status
    # quo it replaces.
    mock: bool = False

    def is_usable(self) -> bool:
        """True only when at least one exchange actually reported."""
        return self.exchange_count > 0 and self.total_balance_coins > 0

    def netflow_usd_1d(self, price_usd: Optional[float]) -> Optional[float]:
        """Convert to USD with a caller-supplied price.

        Returns None when the price is unusable — a netflow without a price
        is not a dollar figure, and fabricating one from a stale or absent
        price is the P169 fee-provenance mistake in another costume.
        """
        if price_usd is None:
            return None
        try:
            p = float(price_usd)
        except (TypeError, ValueError):  # noqa: silent-swallow
            # [P293] No usable price -> no dollar figure (see docstring).
            return None
        if not (p > 0) or p != p:
            return None
        return self.netflow_coins_1d * p

    def balance_pct_1d(self) -> Optional[float]:
        """1d netflow as a fraction of the total exchange balance.

        Price-free and scale-free, so it is comparable across assets and
        across time without any FX assumption.
        """
        if not self.is_usable():
            return None
        return self.netflow_coins_1d / self.total_balance_coins

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_balance_coins": self.total_balance_coins,
            "netflow_coins_1d": self.netflow_coins_1d,
            "netflow_coins_7d": self.netflow_coins_7d,
            "balance_pct_1d": self.balance_pct_1d(),
            "exchange_count": self.exchange_count,
            "mock": self.mock,
            "sign_convention": "positive=inflow_to_exchanges=bearish",
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


@dataclass
class ExchangeNetflowSnapshot:
    timestamp: datetime
    metrics: Dict[str, ExchangeNetflowMetrics] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def get(self, symbol: str) -> Optional[ExchangeNetflowMetrics]:
        return self.metrics.get(symbol.upper())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "errors": list(self.errors),
        }


class ExchangeNetflowFeed:
    """CoinGlass v4 exchange-balance reader. Read-only, fail-soft."""

    V4_BASE_URL = "https://open-api-v4.coinglass.com/api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_sec: float = 3600.0,  # balances update slowly (1d deltas)
        mock_mode: bool = False,
        symbols: tuple = SUPPORTED_SYMBOLS,
    ):
        import os
        self.api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        self.poll_interval_sec = poll_interval_sec
        self._mock_mode = mock_mode or not bool(self.api_key)
        self._symbols = tuple(s.upper() for s in symbols)

        self._last_data: Optional[ExchangeNetflowSnapshot] = None
        self._last_fetch_time: Optional[datetime] = None
        self._fetch_errors = 0
        self._total_fetches = 0

        logger.info(
            f"[EXCH_NETFLOW] Initialized: mock={self._mock_mode} "
            f"symbols={','.join(self._symbols)}"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_latest(self) -> Optional[ExchangeNetflowSnapshot]:
        return self._last_data

    def cache_age_sec(self) -> Optional[float]:
        if self._last_fetch_time is None:
            return None
        _t = self._last_fetch_time
        if _t.tzinfo is None:
            _t = _t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - _t).total_seconds()

    def get_metrics(self, symbol: str) -> Optional[ExchangeNetflowMetrics]:
        """Per-symbol metrics, or None.

        None for SOL is permanent and correct — the endpoint covers no SOL
        exchanges. Callers MUST NOT substitute 0.0.
        """
        if not self._last_data:
            return None
        return self._last_data.get(symbol)

    async def fetch(self) -> Optional[ExchangeNetflowSnapshot]:
        self._total_fetches += 1
        try:
            if self._mock_mode:
                return self._build_mock()
            return await self._fetch_real()
        except Exception as e:
            self._fetch_errors += 1
            logger.warning(f"[EXCH_NETFLOW] Fetch failed: {type(e).__name__}: {e}")
            return self._last_data

    async def fetch_if_stale(self) -> Optional[ExchangeNetflowSnapshot]:
        _age = self.cache_age_sec()
        if _age is not None and _age < self.poll_interval_sec:
            return self._last_data
        return await self.fetch()

    def get_health(self) -> Dict[str, Any]:
        return {
            "source": "coinglass_v4_exchange_balance",
            "mock": self._mock_mode,
            "cache_age_sec": self.cache_age_sec(),
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "symbols_with_data": sorted(
                k for k, v in (self._last_data.metrics.items() if self._last_data else [])
                if v.is_usable()
            ),
        }

    # =========================================================================
    # FETCH
    # =========================================================================

    async def _fetch_real(self) -> ExchangeNetflowSnapshot:
        now = datetime.now(timezone.utc)
        snap = ExchangeNetflowSnapshot(timestamp=now)

        from data_mgmt.feeds._http import create_session

        headers = {"CG-API-KEY": self.api_key, "accept": "application/json"}

        async with create_session() as session:
            for sym in self._symbols:
                try:
                    m = await self._fetch_symbol(session, headers, sym, now)
                    if m is not None and m.is_usable():
                        snap.metrics[sym] = m
                    else:
                        logger.warning(
                            f"[EXCH_NETFLOW] {sym}: endpoint returned no exchange "
                            f"rows — no netflow reading this cycle (NOT zero flow)"
                        )
                except Exception as e:
                    snap.errors.append(f"{sym}:{type(e).__name__}")
                    logger.warning(f"[EXCH_NETFLOW] {sym}: {type(e).__name__}: {e}")

        if not snap.metrics:
            logger.warning(
                "[EXCH_NETFLOW] no usable symbol — keeping previous cache"
            )
            self._fetch_errors += 1
            return self._last_data if self._last_data else snap

        # [P294] Carry the symbols that failed THIS cycle forward from the
        # previous snapshot. Replacing the whole snapshot dropped a
        # still-good reading whenever its sibling failed, and the fetch stamp
        # below then suppressed any retry for a full poll interval — so one
        # transient 500 on ETH erased ETH for an hour. That is the
        # per-family-carry corner P265f/P287 closed for CoinGlass, reopened
        # in a new feed.
        #
        # The carried record keeps its ORIGINAL timestamp, so its age stays
        # honest and a consumer can still tell a fresh reading from a stale
        # one. Carrying is not the same as re-measuring.
        if self._last_data:
            for _sym, _prev in self._last_data.metrics.items():
                if _sym not in snap.metrics:
                    snap.metrics[_sym] = _prev
                    snap.errors.append(f"{_sym}:carried_forward")

        self._last_data = snap
        self._last_fetch_time = now

        logger.info(
            "[EXCH_NETFLOW] "
            + " | ".join(
                f"{k}: net_1d={v.netflow_coins_1d:+,.0f} "
                f"({(v.balance_pct_1d() or 0.0) * 100:+.2f}% of bal, n={v.exchange_count})"
                for k, v in sorted(snap.metrics.items())
            )
            + "  [positive=inflow=bearish]"
        )
        return snap

    async def _fetch_symbol(
        self,
        session: aiohttp.ClientSession,
        headers: Dict[str, str],
        symbol: str,
        now: datetime,
    ) -> Optional[ExchangeNetflowMetrics]:
        url = f"{self.V4_BASE_URL}/exchange/balance/list"
        async with session.get(
            url, headers=headers, params={"symbol": symbol},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status != 200:
                if resp.status == 429:
                    from data_mgmt.feeds._http import parse_retry_after
                    _retry = parse_retry_after(resp.headers.get("Retry-After"))
                    logger.warning(
                        f"[EXCH_NETFLOW] {symbol} rate-limited (429), "
                        f"Retry-After={_retry}s"
                    )
                else:
                    logger.warning(f"[EXCH_NETFLOW] {symbol} -> HTTP {resp.status}")
                return None
            payload = await resp.json()

        # CoinGlass wraps errors in a 200 with a non-zero code.
        if str(payload.get("code", "0")) not in ("0", "200"):
            logger.warning(
                f"[EXCH_NETFLOW] {symbol} API error code={payload.get('code')} "
                f"msg={str(payload.get('msg'))[:120]}"
            )
            return None

        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            return None

        m = ExchangeNetflowMetrics(symbol=symbol, timestamp=now)
        for r in rows:
            if not isinstance(r, dict):
                continue
            m.total_balance_coins += _as_float(r.get("total_balance"))
            m.netflow_coins_1d += _as_float(r.get("balance_change_1d"))
            m.netflow_coins_7d += _as_float(r.get("balance_change_7d"))
            m.exchange_count += 1
        return m

    def _build_mock(self) -> ExchangeNetflowSnapshot:
        now = datetime.now(timezone.utc)
        snap = ExchangeNetflowSnapshot(timestamp=now)
        for sym in self._symbols:
            snap.metrics[sym] = ExchangeNetflowMetrics(
                symbol=sym,
                total_balance_coins=1_000_000.0,
                netflow_coins_1d=0.0,
                netflow_coins_7d=0.0,
                exchange_count=1,
                timestamp=now,
                mock=True,   # [P294] see the field comment
            )
        self._last_data = snap
        self._last_fetch_time = now
        return snap


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return default
        return f
    except (TypeError, ValueError):  # noqa: silent-swallow
        return default


# =============================================================================
# SINGLETON
# =============================================================================

_exchange_netflow_instance: Optional[ExchangeNetflowFeed] = None


def get_exchange_netflow_feed(
    api_key: Optional[str] = None,
    mock_mode: bool = False,
) -> ExchangeNetflowFeed:
    global _exchange_netflow_instance
    if _exchange_netflow_instance is None:
        _exchange_netflow_instance = ExchangeNetflowFeed(
            api_key=api_key, mock_mode=mock_mode
        )
    return _exchange_netflow_instance


def reset_exchange_netflow_feed():
    global _exchange_netflow_instance
    _exchange_netflow_instance = None


if __name__ == "__main__":
    async def _main():
        logging.basicConfig(level=logging.INFO)
        import os
        from pathlib import Path
        if not os.environ.get("COINGLASS_API_KEY"):
            _env = Path(__file__).resolve().parents[2] / ".env"
            if _env.exists():
                for line in _env.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.startswith("COINGLASS_API_KEY="):
                        os.environ["COINGLASS_API_KEY"] = line.split("=", 1)[1].strip()
        feed = ExchangeNetflowFeed()
        snap = await feed.fetch()
        if snap:
            import json
            print(json.dumps(snap.to_dict(), indent=2, default=str))
        print("health:", feed.get_health())

    asyncio.run(_main())

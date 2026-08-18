"""
================================================================================
HMATS - CryptoCompare On-Chain Metrics Feed
================================================================================
Provides: active addresses, transaction count, large transaction (whale proxy)
for BTC and ETH via CryptoCompare /blockchain/latest endpoint.

API Docs: https://min-api.cryptocompare.com/documentation
Auth: api_key query parameter (CRYPTOCOMPARE_API_KEY env var)

Available fields (verified Feb 24):
  active_addresses, transaction_count, large_transaction_count,
  average_transaction_value, new_addresses, hashrate, block_time

NOT available: exchange_inflow_usd, exchange_outflow_usd (need Glassnode/CryptoQuant)
================================================================================
"""

import os
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger("HMATS.CryptoCompareOnChain")

# Attempt aiohttp import
try:
    import aiohttp
    from data_mgmt.feeds._http import create_session, fetch_with_retry
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

BASE_URL = "https://min-api.cryptocompare.com/data"
SUPPORTED_ASSETS = ("BTC", "ETH")
# [P220] 15 min -> 48 h. Each fetch costs 2 calls (BTC + ETH), so at a 4H
# cadence this was 12 calls/day = ~360/month against a 100/MONTH cap. On-chain
# large-transfer counts are a 24h ROLLING statistic, so refreshing them every
# 15 minutes was spending the month to re-read a number that had barely moved.
#
# 48h rather than 24h so total DEMAND lands ON the budget rather than above it:
#   news    12h x 1 call  = 60/month
#   onchain 48h x 2 calls = 30/month
#   total                 = 90 = the budget, under the provider's 100 cap.
# At 24h the total is 120 and the shared guard would bind every month around
# the 3/4 mark — a control that fires routinely stops being read. The guard is
# meant to be a backstop for the unforeseen (another process on the key, a
# manual probe), not the thing that paces us.
MIN_FETCH_INTERVAL = 172800.0  # 48 h
RATE_LIMIT_BACKOFF = 3600.0  # 1 hour backoff on rate limit errors


@dataclass
class CryptoCompareOnChainData:
    """Per-asset on-chain metrics from CryptoCompare blockchain/latest."""
    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Network activity
    active_addresses: int = 0
    new_addresses: int = 0
    transaction_count: int = 0
    average_transaction_value: float = 0.0  # In native units (BTC/ETH)

    # Whale proxy (large transactions > exchange threshold)
    large_transaction_count: int = 0

    # Network health
    hashrate: float = 0.0
    block_time: float = 0.0  # seconds
    current_supply: float = 0.0

    is_mock: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "active_addresses": self.active_addresses,
            "transaction_count": self.transaction_count,
            "large_transaction_count": self.large_transaction_count,
            "average_transaction_value": self.average_transaction_value,
            "hashrate": self.hashrate,
            "is_mock": self.is_mock,
        }


class CryptoCompareOnChainFeed:
    """
    Fetches BTC + ETH on-chain data from CryptoCompare blockchain/latest.

    Rate limited to MIN_FETCH_INTERVAL (120s).
    Falls back to neutral zeros (not random) when API unavailable.
    """

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("CRYPTOCOMPARE_API_KEY", "")
        self._mock_mode = not bool(self._api_key)
        self._last_fetch_time: float = 0.0
        self._data: Dict[str, CryptoCompareOnChainData] = {}
        self._fetch_count: int = 0
        self._error_count: int = 0
        self._rate_limited_until: float = 0.0  # backoff deadline
        # [P253d] Persist the backoff across restarts (the P154/P216 rule,
        # applied to the last CryptoCompare feed that kept it RAM-only). A
        # restart used to disarm an active rate-limit backoff and resume
        # hammering an API that just said "wait" — bounded in the worst case
        # by the persisted P220 account quota, but a limiter that re-arms on
        # restart is not a limiter (P154).
        self._state_path = os.path.join(
            os.environ.get("HMATS_DATA_DIR", "data"), "cc_onchain_state.json")
        self._restore_backoff()

        if self._mock_mode:
            logger.info("[CC_ONCHAIN] MOCK mode (no API key)")
        else:
            logger.info(f"[CC_ONCHAIN] LIVE (key=...{self._api_key[-4:]})")

    def _restore_backoff(self) -> None:
        try:
            import json as _json
            if os.path.exists(self._state_path):
                d = _json.load(open(self._state_path, encoding="utf-8"))
                until = float(d.get("rate_limited_until", 0.0) or 0.0)
                if until > time.time():
                    self._rate_limited_until = until
                    logger.warning(
                        f"[CC_ONCHAIN] restored ACTIVE backoff from disk "
                        f"({until - time.time():.0f}s remaining) — the "
                        f"restart does not reset the limiter")

                # [P293b] Restore the THROTTLE CLOCK and the PAYLOAD too.
                # P253d persisted the backoff and left `_last_fetch_time`
                # (and `_data`) in RAM, one line above its own comment that
                # "a limiter that re-arms on restart is not a limiter". The
                # throttle reads
                #     now - self._last_fetch_time < MIN_FETCH_INTERVAL and self._data
                # so with BOTH halves reset a restart could never be
                # throttled: every process start re-spent 2 calls against a
                # 100/month account cap. Measured: 14 process starts across
                # the retained logs, and the shared budget was exhausted by
                # Aug 10 — nine days into the month.
                #
                # The payload is restored as well, deliberately: restoring
                # only the clock would leave `_data` empty, the `and
                # self._data` clause False, and the persistence decorative —
                # the exact trap P253d called out for the backoff.
                _lft = float(d.get("last_fetch_time", 0.0) or 0.0)
                _payload = d.get("payload") or {}
                if _lft > 0.0 and isinstance(_payload, dict) and _payload:
                    _restored = {}
                    for _sym, _row in _payload.items():
                        if not isinstance(_row, dict):
                            continue
                        try:
                            _ts_raw = _row.get("timestamp")
                            _ts = (datetime.fromisoformat(_ts_raw)
                                   if _ts_raw else datetime.now(timezone.utc))
                            if _ts.tzinfo is None:
                                _ts = _ts.replace(tzinfo=timezone.utc)
                            _restored[_sym] = CryptoCompareOnChainData(
                                symbol=str(_row.get("symbol", _sym)),
                                timestamp=_ts,
                                active_addresses=int(_row.get("active_addresses", 0) or 0),
                                new_addresses=int(_row.get("new_addresses", 0) or 0),
                                transaction_count=int(_row.get("transaction_count", 0) or 0),
                                average_transaction_value=float(
                                    _row.get("average_transaction_value", 0.0) or 0.0),
                                large_transaction_count=int(
                                    _row.get("large_transaction_count", 0) or 0),
                                hashrate=float(_row.get("hashrate", 0.0) or 0.0),
                                block_time=float(_row.get("block_time", 0.0) or 0.0),
                                current_supply=float(_row.get("current_supply", 0.0) or 0.0),
                                is_mock=bool(_row.get("is_mock", True)),
                            )
                        except (TypeError, ValueError):  # noqa: silent-swallow
                            continue  # one corrupt row must not lose the rest
                    if _restored:
                        self._data = _restored
                        self._last_fetch_time = _lft
                        _age_h = (time.time() - _lft) / 3600.0
                        logger.info(
                            f"[CC_ONCHAIN] restored throttle clock + "
                            f"{len(_restored)} asset(s) from disk "
                            f"(age {_age_h:.1f}h of "
                            f"{MIN_FETCH_INTERVAL / 3600.0:.0f}h) — this "
                            f"restart will NOT re-spend account quota")
        except Exception as e:  # noqa: silent-swallow — a corrupt state file must not break startup (P154 precedent); cold start is the stated consequence
            logger.warning(f"[CC_ONCHAIN] backoff restore failed "
                           f"({type(e).__name__}) — cold start")

    def _persist_backoff(self) -> None:
        try:
            import json as _json
            tmp = self._state_path + ".tmp"
            # [P293b] Persist the throttle clock + payload alongside the
            # backoff, so a restart cannot re-spend account quota. Mock rows
            # are never written: restoring them would let a mock survive as
            # if it were a reading (P2).
            _payload = {}
            for _sym, _row in (self._data or {}).items():
                if getattr(_row, "is_mock", True):
                    continue
                _ts = getattr(_row, "timestamp", None)
                _payload[_sym] = {
                    "symbol": getattr(_row, "symbol", _sym),
                    "timestamp": _ts.isoformat() if _ts else None,
                    "active_addresses": getattr(_row, "active_addresses", 0),
                    "new_addresses": getattr(_row, "new_addresses", 0),
                    "transaction_count": getattr(_row, "transaction_count", 0),
                    "average_transaction_value": getattr(
                        _row, "average_transaction_value", 0.0),
                    "large_transaction_count": getattr(
                        _row, "large_transaction_count", 0),
                    "hashrate": getattr(_row, "hashrate", 0.0),
                    "block_time": getattr(_row, "block_time", 0.0),
                    "current_supply": getattr(_row, "current_supply", 0.0),
                    "is_mock": False,
                }
            with open(tmp, "w", encoding="utf-8") as fh:
                _json.dump({
                    "rate_limited_until": self._rate_limited_until,
                    "last_fetch_time": self._last_fetch_time,
                    "payload": _payload,
                }, fh)
            os.replace(tmp, self._state_path)
        except Exception as e:  # noqa: silent-swallow — telemetry persistence must not break the fetch; the RAM value still governs this process
            logger.warning(f"[CC_ONCHAIN] backoff persist failed "
                           f"({type(e).__name__})")

    @property
    def data(self) -> Dict[str, CryptoCompareOnChainData]:
        return self._data

    async def fetch(self) -> Dict[str, CryptoCompareOnChainData]:
        """Fetch on-chain data for BTC and ETH. Rate-limited."""
        if not AIOHTTP_AVAILABLE:
            return self._data

        now = time.time()
        if now - self._last_fetch_time < MIN_FETCH_INTERVAL and self._data:
            return self._data
        # [P253d] Backoff is honored UNCONDITIONALLY. The old `and self._data`
        # meant a restored backoff with an empty post-restart cache did not
        # block anything — the persistence would have been decorative. During
        # a backoff with no cache, serve explicit mocks (the same degradation
        # the in-band rate-limit branch uses) rather than hammering.
        if now < self._rate_limited_until:
            if not self._data:
                for _s in SUPPORTED_ASSETS:
                    self._data[_s] = CryptoCompareOnChainData(symbol=_s, is_mock=True)
            return self._data

        if self._mock_mode:
            for sym in SUPPORTED_ASSETS:
                self._data[sym] = CryptoCompareOnChainData(symbol=sym, is_mock=True)
            self._last_fetch_time = now
            return self._data

        # [P220] Reserve against the SHARED CryptoCompare account budget before
        # spending anything. This feed and cryptocompare_news_feed use the SAME
        # key and the SAME 100-calls/month allowance, but each used to back off
        # independently — so one could exhaust the month while the other kept
        # calling. One fetch here costs len(SUPPORTED_ASSETS) calls, and it is
        # all-or-nothing: half a fetch would spend quota for a partial picture.
        from data_mgmt.feeds._cc_quota import get_cc_quota
        if not get_cc_quota().try_consume(len(SUPPORTED_ASSETS),
                                          caller="cc_onchain"):
            return self._data

        try:
            async with create_session(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                for symbol in SUPPORTED_ASSETS:
                    try:
                        url = f"{BASE_URL}/blockchain/latest"
                        params = {"fsym": symbol, "api_key": self._api_key}

                        async with session.get(
                            url, params=params,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                if resp.status == 429:
                                    from data_mgmt.feeds._http import parse_retry_after
                                    _retry = parse_retry_after(resp.headers.get("Retry-After"))
                                    # [P253d] The 429 branch used to LOG the
                                    # Retry-After and DISCARD it (the P216
                                    # computed-but-unenforced shape) — the
                                    # retry budget the server granted must be
                                    # spent and persisted.
                                    self._rate_limited_until = now + float(
                                        _retry or RATE_LIMIT_BACKOFF)
                                    self._persist_backoff()
                                    logger.warning(
                                        f"[CC_ONCHAIN] {symbol} rate-limited (429), "
                                        f"backing off "
                                        f"{self._rate_limited_until - now:.0f}s "
                                        f"(Retry-After={_retry})"
                                    )
                                else:
                                    logger.warning(f"[CC_ONCHAIN] {symbol}: HTTP {resp.status}")
                                continue

                            raw = await resp.json()
                            block = raw.get("Data", {})

                            # Detect rate limit errors and backoff
                            _resp_msg = str(raw.get("Message", "") or "")
                            if "rate limit" in _resp_msg.lower():
                                self._rate_limited_until = now + RATE_LIMIT_BACKOFF
                                self._persist_backoff()  # [P253d] survive restarts
                                logger.warning(
                                    f"[CC_ONCHAIN] {symbol}: Rate limited — backing off for "
                                    f"{RATE_LIMIT_BACKOFF:.0f}s. Msg: {_resp_msg[:100]}"
                                )
                                # Mark existing data as mock, skip remaining assets
                                for _s in SUPPORTED_ASSETS:
                                    if _s not in self._data:
                                        self._data[_s] = CryptoCompareOnChainData(symbol=_s, is_mock=True)
                                self._last_fetch_time = now
                                return self._data

                            # Diagnostic: log structure if Data key missing or empty
                            if not block:
                                _raw_keys = list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
                                logger.warning(
                                    f"[CC_ONCHAIN] {symbol}: 'Data' key empty/missing in response. "
                                    f"Keys: {_raw_keys}. "
                                    f"Response (truncated): {str(raw)[:300]}"
                                )

                            _active = int(block.get("active_addresses", 0))
                            _new = int(block.get("new_addresses", 0))
                            _tx = int(block.get("transaction_count", 0))
                            _avg_tx = float(block.get("average_transaction_value", 0))
                            _large = int(block.get("large_transaction_count", 0))
                            _hash = float(block.get("hashrate", 0))
                            _bt = float(block.get("block_time", 0))
                            _supply = float(block.get("current_supply", 0))

                            # Mark as mock if all key fields are zero (likely API issue)
                            _all_zero = (_active == 0 and _tx == 0 and _large == 0 and _avg_tx == 0)

                            entry = CryptoCompareOnChainData(
                                symbol=symbol,
                                timestamp=datetime.now(timezone.utc),
                                active_addresses=_active,
                                new_addresses=_new,
                                transaction_count=_tx,
                                average_transaction_value=_avg_tx,
                                large_transaction_count=_large,
                                hashrate=_hash,
                                block_time=_bt,
                                current_supply=_supply,
                                is_mock=_all_zero,
                            )
                            self._data[symbol] = entry

                            if _all_zero:
                                logger.warning(
                                    f"[CC_ONCHAIN] {symbol}: All fields zero ->marked as mock. "
                                    f"Check API key/endpoint. Data keys: {list(block.keys())[:10]}"
                                )
                            else:
                                logger.info(
                                    f"[CC_ONCHAIN] {symbol}: "
                                    f"active_addr={entry.active_addresses:,}, "
                                    f"tx_count={entry.transaction_count:,}, "
                                    f"large_tx={entry.large_transaction_count:,}, "
                                    f"avg_tx_val={entry.average_transaction_value:.4f}"
                                )

                    except Exception as e:
                        logger.warning(f"[CC_ONCHAIN] {symbol} failed: {e}")
                        self._error_count += 1

            self._last_fetch_time = now
            self._fetch_count += 1
            # [P293b] Persist the throttle clock + payload on SUCCESS. Without
            # this the clock is only ever written on a rate-limit path, so a
            # healthy fetch followed by a restart re-spends quota — the very
            # case that exhausted the month's budget in nine days.
            self._persist_backoff()

        except Exception as e:
            logger.warning(f"[CC_ONCHAIN] Session failed: {e}")
            self._error_count += 1

        return self._data

    def get(self, symbol: str) -> Optional[CryptoCompareOnChainData]:
        """Get cached data for a symbol."""
        return self._data.get(symbol.upper())

    def get_status(self) -> Dict[str, Any]:
        return {
            "mock_mode": self._mock_mode,
            "fetch_count": self._fetch_count,
            "error_count": self._error_count,
            "assets": list(self._data.keys()),
            "has_data": any(not d.is_mock for d in self._data.values()),
        }


# Singleton
_instance: Optional[CryptoCompareOnChainFeed] = None


def get_cryptocompare_onchain_feed(**kwargs) -> CryptoCompareOnChainFeed:
    global _instance
    if _instance is None:
        _instance = CryptoCompareOnChainFeed(**kwargs)
    return _instance
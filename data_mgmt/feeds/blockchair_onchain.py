"""[P221] BTC/ETH on-chain metrics from Blockchair — free, keyless, unmetered.

WHY. The CryptoCompare account is capped at 100 calls/MONTH (P220) and the plan
cannot be upgraded, so `/blockchain/latest` is effectively unavailable: it was
the only source of BTC/ETH on-chain metrics, and its absence is why the `flow`
and `onchain` agents read 0.00 on every tick.

Blockchair's `/{chain}/stats` needs **no API key**, covers **both BTC and ETH**,
and was verified live before this module was written:

    bitcoin   transactions_24h 707,682   volume_24h 82,530,807,977,168 sat
    ethereum  transactions_24h 2,344,358 volume_24h_approximate 2.48e24 wei

WHAT IT DOES AND DOES NOT REPLACE. Blockchair does not publish a
"large transaction count", and this module deliberately does NOT invent one —
fabricating a number to satisfy a downstream `> 0` check is precisely the class
of defect this codebase keeps finding. What it publishes instead is the honest
quantity Blockchair actually has:

    onchain_volume_24h_usd    total on-chain settlement value
    largest_transaction_usd   the single biggest transfer in 24h
    transaction_count / average_transaction_value

and `volume_24h` is arguably a *better* whale magnitude than CryptoCompare's
`large_tx_count x avg_tx_value`, which was itself an estimate (its own comment
said "we don't have price here, so use count as the signal").

IT IS A MAGNITUDE, NOT A DIRECTION — and that is unchanged from before. The
whale proxy's *sign* has always come from CoinGlass OI + funding
(`main.py:7263`), never from the on-chain feed; CryptoCompare only ever supplied
a size. So replacing the size source loses nothing directional.

UNITS. Blockchair returns native base units (satoshi, wei) and a
`market_price_usd` in the same payload, so USD conversion happens here rather
than being left to a caller that would have to guess the denomination — the
P169 lesson that a value without its unit is not a measurement.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("HMATS.BlockchairOnChain")

BASE_URL = "https://api.blockchair.com"
# chain path -> (asset, base units per coin)
CHAINS = {"bitcoin": ("BTC", 1e8), "ethereum": ("ETH", 1e18)}

# Free and unmetered, but not a licence to hammer: these are 24h rolling
# statistics, so anything under an hour re-reads a number that has not moved.
MIN_FETCH_INTERVAL = 3600.0  # 1 h
HTTP_TIMEOUT = 15.0


@dataclass
class BlockchairOnChainData:
    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transaction_count: int = 0            # transactions_24h
    average_transaction_value: float = 0.0  # native units, matching the CC field
    onchain_volume_24h_usd: float = 0.0
    largest_transaction_usd: float = 0.0
    market_price_usd: float = 0.0
    mempool_transactions: int = 0
    is_mock: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "transaction_count": self.transaction_count,
            "average_transaction_value": self.average_transaction_value,
            "onchain_volume_24h_usd": self.onchain_volume_24h_usd,
            "largest_transaction_usd": self.largest_transaction_usd,
            "market_price_usd": self.market_price_usd,
            "is_mock": self.is_mock,
        }


def _f(v: Any, default: float = 0.0) -> float:
    """Coerce a JSON scalar to float. Blockchair returns wei as a STRING, and
    absent fields as null, so both shapes are expected rather than exceptional.
    """
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):  # noqa: silent-swallow — a malformed scalar is a data shape, not an error; logging every one would drown the feed
        return default


class BlockchairOnChainFeed:
    """Keyless BTC/ETH on-chain stats. Never raises to the caller."""

    def __init__(self) -> None:
        self._data: Dict[str, BlockchairOnChainData] = {}
        self._last_fetch_time: float = 0.0
        self._error_count = 0
        self._disabled = os.environ.get("HMATS_DISABLE_BLOCKCHAIR", "") == "1"
        if self._disabled:
            logger.info("[BLOCKCHAIR] disabled via HMATS_DISABLE_BLOCKCHAIR=1")
        else:
            logger.info("[BLOCKCHAIR] LIVE (keyless, unmetered)")

    @property
    def data(self) -> Dict[str, BlockchairOnChainData]:
        return self._data

    def _fetch_chain(self, chain: str) -> Optional[BlockchairOnChainData]:
        asset, base_units = CHAINS[chain]
        req = urllib.request.Request(
            f"{BASE_URL}/{chain}/stats",
            headers={"User-Agent": "hmats/1.0"},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            payload = json.load(resp)
        d = payload.get("data") or {}
        if not d:
            return None

        price = _f(d.get("market_price_usd"))
        txs = int(_f(d.get("transactions_24h")))
        # BTC reports `volume_24h`; ETH reports `volume_24h_approximate` (a
        # string, because wei overflows JSON's safe integer range).
        vol_native = _f(d.get("volume_24h")) or _f(d.get("volume_24h_approximate"))
        vol_coins = vol_native / base_units if base_units else 0.0

        # Blockchair reports this ALREADY IN USD as `value_usd` — verified
        # against the live payload. Reading `value` (the natural guess, and my
        # first version) silently produced 0.0 on both chains, which would have
        # shipped a whale metric that is always zero: exactly the reader/writer
        # key mismatch this codebase keeps finding (P2/P197's `entry_vwap`).
        largest = d.get("largest_transaction_24h") or {}
        largest_usd = _f(largest.get("value_usd") if isinstance(largest, dict) else 0.0)

        return BlockchairOnChainData(
            symbol=asset,
            transaction_count=txs,
            # Native units per transaction — same semantics as the CryptoCompare
            # field this replaces, so downstream arithmetic is unchanged.
            average_transaction_value=(vol_coins / txs) if txs > 0 else 0.0,
            onchain_volume_24h_usd=vol_coins * price,
            largest_transaction_usd=largest_usd,
            market_price_usd=price,
            mempool_transactions=int(_f(d.get("mempool_transactions"))),
            is_mock=False,
        )

    def fetch(self) -> Dict[str, BlockchairOnChainData]:
        """Refresh both chains. Synchronous (stdlib urllib) and cheap: two
        requests an hour against an unmetered endpoint."""
        if self._disabled:
            return self._data
        now = time.time()
        if self._data and (now - self._last_fetch_time) < MIN_FETCH_INTERVAL:
            return self._data

        got_any = False
        for chain in CHAINS:
            asset = CHAINS[chain][0]
            try:
                rec = self._fetch_chain(chain)
                if rec is not None:
                    self._data[asset] = rec
                    got_any = True
            except Exception as e:  # noqa: silent-swallow — one chain must not kill the other
                self._error_count += 1
                logger.warning(
                    f"[BLOCKCHAIR] {asset} fetch failed: {type(e).__name__}: {e}"
                )

        if got_any:
            # Only advance the clock on a real refresh, so a failure retries on
            # the next tick instead of being cached out for an hour.
            self._last_fetch_time = now
            logger.info(
                "[BLOCKCHAIR] "
                + " | ".join(
                    f"{a}: txs={r.transaction_count:,} "
                    f"vol24h=${r.onchain_volume_24h_usd:,.0f}"
                    for a, r in sorted(self._data.items())
                )
            )
        return self._data

    def get_status(self) -> Dict[str, Any]:
        return {
            "disabled": self._disabled,
            "assets": sorted(self._data),
            "errors": self._error_count,
            "age_sec": round(time.time() - self._last_fetch_time, 1)
            if self._last_fetch_time else None,
        }


_FEED: Optional[BlockchairOnChainFeed] = None


def get_blockchair_onchain_feed() -> BlockchairOnChainFeed:
    global _FEED
    if _FEED is None:
        _FEED = BlockchairOnChainFeed()
    return _FEED

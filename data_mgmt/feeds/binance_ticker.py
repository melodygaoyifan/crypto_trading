"""Binance public ticker + orderbook fetcher — no API key required.

Populates market_data keys consumed by microstructure_agent:
  - binance_bid, binance_ask
  - binance_bid_size, binance_ask_size
  - binance_taker_buy_volume, binance_taker_sell_volume

Uses Binance REST API (/api/v3/*). Async-safe. Graceful failures.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from data_mgmt.feeds._http import create_session

logger = logging.getLogger("BinanceTicker")

_SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

_BASE = "https://api.binance.com"


async def fetch_binance_snapshot(asset: str, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
    """Fetch best bid/ask + top-of-book sizes + 1h taker flows.

    Returns dict with keys:
      binance_bid, binance_ask, binance_bid_size, binance_ask_size,
      binance_taker_buy_volume, binance_taker_sell_volume,
      binance_last_price, binance_taker_flow_valid

    [P253] binance_timestamp_ms was promised here and never returned — the
    docstring lied for the feed's whole life. And taker_buy/sell were 0.0 on
    a kline failure with nothing marking them, so a failed fetch was
    byte-identical to genuinely zero flow (the P199/P216 conflation).
    binance_taker_flow_valid distinguishes the two: False means the zeros
    are an OUTAGE, not a reading.

    Returns None on failure.
    """
    sym = _SYMBOL_MAP.get(asset.upper())
    if not sym:
        return None

    try:
        async with create_session() as session:
            # Book ticker: best bid/ask + sizes
            book_url = f"{_BASE}/api/v3/ticker/bookTicker?symbol={sym}"
            # 24h ticker: last price + 24h taker ratios (approximation of flow)
            stats_url = f"{_BASE}/api/v3/ticker/24hr?symbol={sym}"

            async def _get(url):
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None
                    return await resp.json()

            book, stats = await asyncio.gather(_get(book_url), _get(stats_url))
            if not book:
                return None

            bid = float(book.get("bidPrice", 0) or 0)
            ask = float(book.get("askPrice", 0) or 0)
            bid_sz = float(book.get("bidQty", 0) or 0)
            ask_sz = float(book.get("askQty", 0) or 0)

            # Binance 24hr ticker provides: volume (quote), count (trades), but no split.
            # Use klines endpoint (1h) for taker-buy volume split — only pull latest 1h bar.
            taker_buy = 0.0
            taker_sell = 0.0
            taker_flow_valid = False  # [P253] failure and zero must differ
            last_price = float(stats.get("lastPrice", 0) or 0) if stats else 0.0
            try:
                kline_url = f"{_BASE}/api/v3/klines?symbol={sym}&interval=1h&limit=1"
                k = await _get(kline_url)
                if k and len(k) > 0:
                    # Binance kline: [openTime, open, high, low, close, volume, closeTime,
                    #                 quoteVolume, numTrades, takerBuyBase, takerBuyQuote, ignore]
                    bar = k[0]
                    total_vol = float(bar[5] or 0)
                    taker_buy = float(bar[9] or 0)
                    taker_sell = max(0.0, total_vol - taker_buy)
                    taker_flow_valid = True
            except Exception as _kline_err:
                # [P22 2026-04-24] was bare except: pass which silently returned
                # taker_buy=taker_sell=0 as if it were valid data, feeding the
                # microstructure agent zeroed flow on every Binance kline failure.
                # Log + leave at 0 so the failure is visible in heartbeat/logs.
                logger.debug(
                    f"[BINANCE_TICKER] {asset} kline fetch failed: {_kline_err!r}; "
                    f"taker flow unavailable this tick"
                )

            return {
                "binance_bid": bid,
                "binance_ask": ask,
                "binance_bid_size": bid_sz,
                "binance_ask_size": ask_sz,
                "binance_taker_buy_volume": taker_buy,
                "binance_taker_sell_volume": taker_sell,
                "binance_last_price": last_price,
                "binance_taker_flow_valid": taker_flow_valid,
            }
    except asyncio.TimeoutError:
        logger.debug(f"[BINANCE_TICKER] {asset} timeout")
    except Exception as e:
        logger.debug(f"[BINANCE_TICKER] {asset} error: {e}")
    return None


async def fetch_all_assets(assets: list) -> Dict[str, Dict[str, Any]]:
    """Fetch snapshots for multiple assets in parallel."""
    results = await asyncio.gather(
        *(fetch_binance_snapshot(a) for a in assets),
        return_exceptions=True,
    )
    out = {}
    for asset, res in zip(assets, results):
        if isinstance(res, dict):
            out[asset] = res
    return out

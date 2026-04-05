"""
HMATS - Orderbook Snapshot Collector
=====================================

Collects 25-level orderbook depth from Kraken REST API every 60 seconds.
Stores as JSONL in data/orderbook_snapshots/{ASSET}_{date}.jsonl

Usage:
    python -X utf8 scripts/collect_orderbook.py
    python -X utf8 scripts/collect_orderbook.py --assets BTC ETH --interval 30
    python -X utf8 scripts/collect_orderbook.py --duration 3600  # 1 hour
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp

logger = logging.getLogger("collect_orderbook")

# Kraken REST API
KRAKEN_REST_URL = "https://api.kraken.com/0/public/Depth"

# Kraken pair mapping
PAIR_MAP = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "SOL": "SOLUSD",
}

OUTPUT_DIR = Path("data/orderbook_snapshots")

_running = True


def _handle_signal(signum, frame):
    global _running
    _running = False
    logger.info(f"Received signal {signum}, shutting down...")


async def fetch_orderbook(session: aiohttp.ClientSession, asset: str, depth: int = 25):
    """Fetch orderbook from Kraken REST API."""
    pair = PAIR_MAP.get(asset.upper())
    if not pair:
        logger.warning(f"Unknown asset: {asset}")
        return None

    params = {"pair": pair, "count": depth}
    try:
        async with session.get(KRAKEN_REST_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.warning(f"[{asset}] HTTP {resp.status}")
                return None
            data = await resp.json()
            if data.get("error"):
                logger.warning(f"[{asset}] API error: {data['error']}")
                return None

            # Kraken returns under the pair key
            result = data.get("result", {})
            book = None
            for key in result:
                book = result[key]
                break

            if not book:
                return None

            ts = datetime.now(timezone.utc).isoformat()
            return {
                "timestamp": ts,
                "asset": asset.upper(),
                "asks": [[float(p), float(v), int(t)] for p, v, t in book.get("asks", [])],
                "bids": [[float(p), float(v), int(t)] for p, v, t in book.get("bids", [])],
                "depth_levels": depth,
                "ask_depth_usd": sum(float(p) * float(v) for p, v, _ in book.get("asks", [])),
                "bid_depth_usd": sum(float(p) * float(v) for p, v, _ in book.get("bids", [])),
                "mid_price": (float(book["asks"][0][0]) + float(book["bids"][0][0])) / 2
                             if book.get("asks") and book.get("bids") else 0.0,
                "spread_bps": ((float(book["asks"][0][0]) - float(book["bids"][0][0]))
                               / float(book["bids"][0][0]) * 10000)
                              if book.get("asks") and book.get("bids") else 0.0,
            }
    except asyncio.TimeoutError:
        logger.warning(f"[{asset}] Timeout")
        return None
    except Exception as e:
        logger.warning(f"[{asset}] Error: {e}")
        return None


def write_snapshot(snapshot: dict):
    """Append snapshot to daily JSONL file."""
    asset = snapshot["asset"]
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    filepath = OUTPUT_DIR / f"{asset}_{date_str}.jsonl"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")


async def collect_loop(assets: list, interval: float, duration: float):
    """Main collection loop."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    snap_count = 0

    logger.info(f"Starting orderbook collection: assets={assets}, interval={interval}s, "
                f"duration={'unlimited' if duration <= 0 else f'{duration}s'}")

    async with aiohttp.ClientSession() as session:
        while _running:
            if 0 < duration < (time.time() - start_time):
                logger.info(f"Duration {duration}s reached, stopping.")
                break

            tasks = [fetch_orderbook(session, asset) for asset in assets]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    logger.warning(f"Fetch error: {res}")
                    continue
                if res is not None:
                    write_snapshot(res)
                    snap_count += 1
                    if snap_count % 10 == 0:
                        logger.info(
                            f"[{res['asset']}] mid={res['mid_price']:.2f} "
                            f"spread={res['spread_bps']:.1f}bps "
                            f"ask_depth=${res['ask_depth_usd']:,.0f} "
                            f"bid_depth=${res['bid_depth_usd']:,.0f}"
                        )

            await asyncio.sleep(interval)

    logger.info(f"Collection complete. {snap_count} snapshots written to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Kraken Orderbook Collector")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL"],
                        help="Assets to collect (default: BTC ETH SOL)")
    parser.add_argument("--interval", type=float, default=60.0,
                        help="Collection interval in seconds (default: 60)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Total duration in seconds (0=unlimited, default: 0)")
    parser.add_argument("--depth", type=int, default=25,
                        help="Orderbook depth levels (default: 25)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    asyncio.run(collect_loop(args.assets, args.interval, args.duration))


if __name__ == "__main__":
    main()

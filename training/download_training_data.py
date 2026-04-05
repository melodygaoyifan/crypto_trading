#!/usr/bin/env python3
"""
================================================================================
HMATS Training 数据下载脚本 v2
================================================================================

数据频率:
- 现货 OHLCV: 1H (小时线) - 主训练数据
- 期货 OHLCV: 1D (日线)
- Funding Rate: 1D (日线)
- Open Interest: 1D (日线)
- 波动率: 1D (日线)

用法:
    python download_training_data.py --token YOUR_API_TOKEN --years 5
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Training-Data")

API_BASE = "https://api.cryptodatadownload.com/v1"

ASSETS = {
    'SOL': {'symbol': 'SOLUSDT', 'launch_date': '2020-08-11'},
    'BTC': {'symbol': 'BTCUSDT', 'launch_date': '2017-08-17'},
    'ETH': {'symbol': 'ETHUSDT', 'launch_date': '2017-08-17'},
}


class CDDClient:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
        })
        self.request_count = 0
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        time.sleep(1)  # Rate limit
        url = API_BASE + endpoint
        self.request_count += 1
        
        try:
            resp = self.session.get(url, params=params, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(60)
                return self._request(endpoint, params)
            else:
                logger.warning(f"[WARN]️ {resp.status_code}: {endpoint}")
                return None
        except Exception as e:
            logger.error(f"OFF {e}")
            return None
    
    def _parse(self, data) -> Optional[pd.DataFrame]:
        if data is None:
            return None
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            return None
        if len(df) == 0:
            return None
        df.columns = df.columns.str.lower()
        return df
    
    def get_spot_1h(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """现货 OHLCV (1H)"""
        data = self._request(f"/ohlc/binance/spot/{symbol}", {'timeframe': '1h', 'start': start, 'end': end})
        return self._parse(data)
    
    def get_futures_1d(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """期货 OHLCV (1D 日线)"""
        for endpoint in [f"/ohlc/binance/futures/{symbol}", f"/ohlc/binance/futures/um/{symbol}"]:
            data = self._request(endpoint, {'timeframe': '1d', 'start': start, 'end': end})
            df = self._parse(data)
            if df is not None:
                return df
        return None
    
    def get_funding_1d(self, symbol: str, start: str) -> Optional[pd.DataFrame]:
        """Funding Rate (日线)"""
        for endpoint in [f"/data/binance/futures/{symbol}/funding", f"/summary/binance/futures/funding/{symbol}"]:
            data = self._request(endpoint, {'start': start})
            df = self._parse(data)
            if df is not None:
                return df
        return None
    
    def get_oi_1d(self, symbol: str, start: str) -> Optional[pd.DataFrame]:
        """Open Interest (日线)"""
        for endpoint in [f"/data/binance/futures/{symbol}/oi", f"/summary/binance/futures/metrics/{symbol}"]:
            data = self._request(endpoint, {'start': start})
            df = self._parse(data)
            if df is not None:
                return df
        return None
    
    def get_volatility_1d(self, symbol: str) -> Optional[pd.DataFrame]:
        """波动率 (日线)"""
        data = self._request(f"/summary/deribit/volatility/{symbol}")
        return self._parse(data)


class Downloader:
    def __init__(self, client: CDDClient, output_dir: str = 'training_data'):
        self.client = client
        self.output_dir = Path(output_dir)
        for d in ['raw', 'futures_daily', 'funding_daily', 'oi_daily', 'volatility_daily']:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)
    
    def _dates(self, asset: str, years: int):
        config = ASSETS.get(asset, {})
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        launch = config.get('launch_date', '2017-01-01')
        if start < launch:
            start = launch
        return start, end
    
    def download_all(self, assets: List[str], years: int = 5):
        logger.info("=" * 60)
        logger.info(f"🎓 HMATS Training 数据下载")
        logger.info(f"   资产: {assets}, 年数: {years}")
        logger.info("   现货=1H, 其他=1D (日线)")
        logger.info("=" * 60)
        
        results = {}
        
        for asset in assets:
            logger.info(f"\n📦 {asset}")
            config = ASSETS.get(asset)
            if not config:
                continue
            
            symbol = config['symbol']
            start, end = self._dates(asset, years)
            
            # 现货 1H
            logger.info(f"  📥 现货 OHLCV [1H]...")
            df = self.client.get_spot_1h(symbol, start, end)
            if df is not None:
                path = self.output_dir / 'raw' / f'{asset}_60m.parquet'
                df.to_parquet(path)
                logger.info(f"    ON {len(df):,} 行")
                results[f'{asset}_spot'] = len(df)
            
            # 期货 1D
            logger.info(f"  📥 期货 OHLCV [1D 日线]...")
            df = self.client.get_futures_1d(symbol, max(start, '2019-09-01'), end)
            if df is not None:
                path = self.output_dir / 'futures_daily' / f'{asset}_futures_1d.parquet'
                df.to_parquet(path)
                logger.info(f"    ON {len(df):,} 行 (日线)")
                results[f'{asset}_futures'] = len(df)
            
            # Funding 1D
            logger.info(f"  📥 Funding Rate [1D 日线]...")
            df = self.client.get_funding_1d(symbol, start)
            if df is not None:
                path = self.output_dir / 'funding_daily' / f'{asset}_funding_1d.parquet'
                df.to_parquet(path)
                logger.info(f"    ON {len(df):,} 行 (日线)")
                results[f'{asset}_funding'] = len(df)
            
            # OI 1D
            logger.info(f"  📥 Open Interest [1D 日线]...")
            df = self.client.get_oi_1d(symbol, start)
            if df is not None:
                path = self.output_dir / 'oi_daily' / f'{asset}_oi_1d.parquet'
                df.to_parquet(path)
                logger.info(f"    ON {len(df):,} 行 (日线)")
                results[f'{asset}_oi'] = len(df)
        
        # 波动率
        logger.info(f"\n📦 波动率")
        for symbol in ['BTC', 'ETH']:
            logger.info(f"  📥 {symbol} 波动率 [1D 日线]...")
            df = self.client.get_volatility_1d(symbol)
            if df is not None:
                path = self.output_dir / 'volatility_daily' / f'{symbol}_volatility_1d.parquet'
                df.to_parquet(path)
                logger.info(f"    ON {len(df):,} 行 (日线)")
                results[f'{symbol}_vol'] = len(df)
        
        # 摘要
        logger.info("\n" + "=" * 60)
        logger.info(f"📊 总计: {sum(results.values()):,} 行, API请求: {self.client.request_count}")
        logger.info("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='HMATS Training 数据下载')
    parser.add_argument('--token', type=str, required=True, help='API Token')
    parser.add_argument('--years', type=int, default=5, help='年数')
    parser.add_argument('--assets', nargs='+', default=['SOL', 'BTC', 'ETH'])
    parser.add_argument('--output', type=str, default='training_data')
    args = parser.parse_args()
    
    logger.info(f"🔑 Token: {args.token[:8]}...")
    
    client = CDDClient(args.token)
    downloader = Downloader(client, args.output)
    downloader.download_all(args.assets, args.years)
    
    logger.info("\nON 下载完成!")
    logger.info("下一步: python run_training_full.py --epochs 100")


if __name__ == "__main__":
    main()

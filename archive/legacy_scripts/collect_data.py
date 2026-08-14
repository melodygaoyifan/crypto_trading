#!/usr/bin/env python3
"""
数据收集脚本 - 支持多种数据源

数据源:
1. Kraken 官方历史数据 (推荐) - 免费完整历史
2. CryptoDataDownload - 免费 CSV
3. Binance API - 免费 API
4. Kaggle 数据集 - 需要 Kaggle 账号

用法:
    python collect_data.py --source kraken --pairs BTC ETH SOL --output ./training_data/raw
    python collect_data.py --source binance --pairs BTC ETH SOL --start 2023-01-01
    python collect_data.py --source kaggle --dataset mczielinski/bitcoin-historical-data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import requests
import zipfile
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("DataCollector")

# =============================================================================
# 数据源配置
# =============================================================================

DATA_SOURCES = {
    'kraken': {
        'url': 'https://support.kraken.com/articles/360047124832-downloadable-historical-ohlcvt-open-high-low-close-volume-trades-data',
        'direct_download': 'https://download.kraken.com/ohlcvt/historical/OHLCVT_all.zip',
        'description': 'Kraken 官方完整历史数据 (推荐)',
        'coverage': '从上市至今',
        'format': 'CSV in ZIP',
    },
    'cryptodatadownload': {
        'base_url': 'https://www.cryptodatadownload.com/cdd/',
        'description': 'CryptoDataDownload 免费 CSV',
        'coverage': '2017-至今',
        'format': 'CSV',
    },
    'binance': {
        'api_url': 'https://api.binance.com/api/v3/klines',
        'description': 'Binance API (需要翻墙)',
        'coverage': '2017-至今',
        'format': 'JSON',
    },
    'kaggle': {
        'datasets': {
            'btc_minute': 'mczielinski/bitcoin-historical-data',
            'btc_eth': 'kapturovalexander/bitcoin-and-ethereum-prices-from-start-to-2023',
            'sol': 'heidarmirhajisadati/solana-historical-data-2021-2024',
            'multi_crypto': 'franoisgeorgesjulien/crypto',
        },
        'description': 'Kaggle 数据集 (需要 Kaggle 账号)',
    },
}


# =============================================================================
# CryptoDataDownload 收集器
# =============================================================================

class CryptoDataDownloadCollector:
    """从 CryptoDataDownload 下载数据"""
    
    # 直接下载链接
    DOWNLOAD_URLS = {
        'BTC': {
            'binance': 'https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv',
            'kraken': 'https://www.cryptodatadownload.com/cdd/Kraken_BTCUSD_1h.csv',
            'gemini': 'https://www.cryptodatadownload.com/cdd/Gemini_BTCUSD_1h.csv',
        },
        'ETH': {
            'binance': 'https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_1h.csv',
            'kraken': 'https://www.cryptodatadownload.com/cdd/Kraken_ETHUSD_1h.csv',
            'gemini': 'https://www.cryptodatadownload.com/cdd/Gemini_ETHUSD_1h.csv',
        },
        'SOL': {
            'binance': 'https://www.cryptodatadownload.com/cdd/Binance_SOLUSDT_1h.csv',
            'kraken': 'https://www.cryptodatadownload.com/cdd/Kraken_SOLUSD_1h.csv',
        },
    }
    
    def __init__(self, output_dir: str = './training_data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, asset: str, exchange: str = 'binance') -> pd.DataFrame:
        """下载指定资产的数据"""
        if asset not in self.DOWNLOAD_URLS:
            logger.error(f"不支持的资产: {asset}")
            return None
        
        if exchange not in self.DOWNLOAD_URLS[asset]:
            exchange = list(self.DOWNLOAD_URLS[asset].keys())[0]
            logger.warning(f"使用默认交易所: {exchange}")
        
        url = self.DOWNLOAD_URLS[asset][exchange]
        logger.info(f"下载 {asset} 数据从 {exchange}...")
        
        try:
            # 下载
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # 解析 CSV (跳过第一行注释)
            lines = response.text.split('\n')
            if lines[0].startswith('http') or 'cryptodatadownload' in lines[0].lower():
                lines = lines[1:]
            
            from io import StringIO
            df = pd.read_csv(StringIO('\n'.join(lines)))
            
            # 标准化列名
            df = self._standardize_columns(df)
            
            logger.info(f"✓ {asset}: {len(df)} 条数据, {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 常见列名映射
        column_map = {
            'date': 'timestamp',
            'Date': 'timestamp',
            'Unix': 'unix',
            'unix': 'unix',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Volume BTC': 'volume',
            'Volume USD': 'volume_usd',
            'tradecount': 'count',
        }
        
        df = df.rename(columns=column_map)
        
        # 处理时间戳
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'unix' in df.columns:
            df['timestamp'] = pd.to_datetime(df['unix'], unit='s')
        
        # 确保必要列存在
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df['volume'] = 0
                else:
                    logger.warning(f"缺少列: {col}")
        
        # 排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    def download_all(self, assets: list = ['BTC', 'ETH', 'SOL'], exchange: str = 'binance'):
        """下载所有资产"""
        results = {}
        for asset in assets:
            df = self.download(asset, exchange)
            if df is not None:
                # 保存
                output_path = self.output_dir / f"{asset}_60m.parquet"
                df.to_parquet(output_path)
                logger.info(f"✓ 保存: {output_path}")
                results[asset] = df
        
        return results


# =============================================================================
# Binance API 收集器
# =============================================================================

class BinanceCollector:
    """从 Binance API 收集数据"""
    
    API_URL = 'https://api.binance.com/api/v3/klines'
    
    SYMBOLS = {
        'BTC': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'SOL': 'SOLUSDT',
    }
    
    def __init__(self, output_dir: str = './training_data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, asset: str, start_date: str = '2023-01-01', 
                 end_date: str = None, interval: str = '1h') -> pd.DataFrame:
        """下载数据"""
        if asset not in self.SYMBOLS:
            logger.error(f"不支持的资产: {asset}")
            return None
        
        symbol = self.SYMBOLS[asset]
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date or datetime.now()).timestamp() * 1000)
        
        logger.info(f"下载 {asset} ({symbol}) 从 {start_date}...")
        
        all_data = []
        current_start = start_ts
        
        while current_start < end_ts:
            try:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': 1000,
                }
                
                response = requests.get(self.API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][0] + 1
                
                logger.info(f"  获取 {len(data)} 条, 总计 {len(all_data)} 条")
                time.sleep(0.5)  # 限速
                
            except Exception as e:
                logger.error(f"API 错误: {e}")
                break
        
        if not all_data:
            return None
        
        # 转换为 DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
            'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
        })
        
        logger.info(f"✓ {asset}: {len(df)} 条数据")
        
        return df
    
    def download_all(self, assets: list = ['BTC', 'ETH', 'SOL'], 
                     start_date: str = '2023-01-01'):
        """下载所有资产"""
        results = {}
        for asset in assets:
            df = self.download(asset, start_date)
            if df is not None:
                output_path = self.output_dir / f"{asset}_60m.parquet"
                df.to_parquet(output_path)
                logger.info(f"✓ 保存: {output_path}")
                results[asset] = df
        
        return results


# =============================================================================
# Kaggle 数据集下载器
# =============================================================================

class KaggleDownloader:
    """从 Kaggle 下载数据集"""
    
    DATASETS = {
        'btc_minute': {
            'id': 'mczielinski/bitcoin-historical-data',
            'description': 'BTC 分钟级数据 (2012-2021)',
            'file': 'bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv',
        },
        'btc_eth_2023': {
            'id': 'kapturovalexander/bitcoin-and-ethereum-prices-from-start-to-2023',
            'description': 'BTC/ETH 数据 (2014-2023)',
        },
        'sol_2024': {
            'id': 'heidarmirhajisadati/solana-historical-data-2021-2024',
            'description': 'SOL 数据 (2021-2024)',
        },
        'crypto_hourly': {
            'id': 'franoisgeorgesjulien/crypto',
            'description': '多币种小时数据 (2017-2023)',
        },
        'crypto_futures': {
            'id': 'arthurneuron/cryptocurrency-futures-ohlcv-dataset-1m-2024',
            'description': '期货 OHLCV 分钟数据 (2024)',
        },
    }
    
    def __init__(self, output_dir: str = './training_data/kaggle'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, dataset_key: str):
        """下载 Kaggle 数据集"""
        if dataset_key not in self.DATASETS:
            logger.error(f"未知数据集: {dataset_key}")
            return None
        
        dataset = self.DATASETS[dataset_key]
        dataset_id = dataset['id']
        
        logger.info(f"下载 Kaggle 数据集: {dataset_id}")
        logger.info(f"描述: {dataset['description']}")
        
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(self.output_dir / dataset_key),
                unzip=True
            )
            logger.info(f"✓ 下载完成: {self.output_dir / dataset_key}")
            return True
        except ImportError:
            logger.error("需要安装 kaggle: pip install kaggle")
            logger.info("并配置 ~/.kaggle/kaggle.json")
            return False
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False
    
    def list_datasets(self):
        """列出可用数据集"""
        print("\n可用的 Kaggle 数据集:")
        print("-" * 60)
        for key, info in self.DATASETS.items():
            print(f"  {key}: {info['description']}")
            print(f"    ID: {info['id']}")
        print("-" * 60)


# =============================================================================
# 数据处理
# =============================================================================

def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """将分钟数据重采样为小时数据"""
    df = df.set_index('timestamp')
    
    resampled = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()
    
    return resampled.reset_index()


def merge_datasets(dfs: list, name: str = 'merged') -> pd.DataFrame:
    """合并多个数据集"""
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    
    # 填充缺失值
    merged = merged.set_index('timestamp')
    merged = merged.resample('1H').ffill()
    
    logger.info(f"合并后: {len(merged)} 条数据")
    return merged.reset_index()


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='数据收集脚本')
    parser.add_argument('--source', type=str, default='cryptodatadownload',
                       choices=['cryptodatadownload', 'binance', 'kaggle', 'list'],
                       help='数据源')
    parser.add_argument('--pairs', nargs='+', default=['BTC', 'ETH', 'SOL'],
                       help='交易对')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='交易所 (cryptodatadownload)')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='开始日期 (binance)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Kaggle 数据集 key')
    parser.add_argument('--output', type=str, default='./training_data/raw',
                       help='输出目录')
    parser.add_argument('--list-kaggle', action='store_true',
                       help='列出 Kaggle 数据集')
    
    args = parser.parse_args()
    
    # 列出 Kaggle 数据集
    if args.list_kaggle or args.source == 'list':
        KaggleDownloader().list_datasets()
        return
    
    logger.info("="*60)
    logger.info("数据收集脚本")
    logger.info("="*60)
    
    if args.source == 'cryptodatadownload':
        collector = CryptoDataDownloadCollector(args.output)
        collector.download_all(args.pairs, args.exchange)
    
    elif args.source == 'binance':
        collector = BinanceCollector(args.output)
        collector.download_all(args.pairs, args.start)
    
    elif args.source == 'kaggle':
        downloader = KaggleDownloader()
        if args.dataset:
            downloader.download(args.dataset)
        else:
            downloader.list_datasets()
    
    logger.info("="*60)
    logger.info("完成!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
================================================================================
HMATS Training v3.1 - 完整数据获取脚本
================================================================================

数据需求:
1. OHLCV 价格数据 (BTC/ETH/SOL) - DT v3.2, DRL v5.5
   - 最少 8,000+ 行 (约 1 年)
   - 推荐 20,000+ 行 (约 2.3 年)
   
2. 情感数据 (新闻/推文) - Sentiment v2.2
   - 最少 5,000+ 条
   - 需要标签 (positive/neutral/negative)

数据源:
- OHLCV: Kraken API, Binance Data Vision, CryptoDataDownload
- 情感: CryptoNews API, Twitter API, Kaggle 数据集

用法:
    # 下载全部 OHLCV 数据
    python get_data.py --ohlcv
    
    # 下载情感数据
    python get_data.py --sentiment
    
    # 全部下载
    python get_data.py --all
    
    # 使用 Kraken API
    python get_data.py --ohlcv --source kraken
    
    # 从本地文件导入
    python get_data.py --import-csv path/to/data.csv --asset SOL

================================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("DataCollector")


# =============================================================================
# 数据需求配置
# =============================================================================

DATA_REQUIREMENTS = {
    'DT_v32': {
        'min_rows': 5000,      # 最少 ~208 天
        'recommended': 15000,  # 推荐 ~625 天
        'context': 96,         # 上下文窗口
        'features': 80,        # 特征维度
    },
    'DRL_v55': {
        'min_rows': 3000,      # 最少 ~125 天
        'recommended': 10000,  # 推荐 ~416 天
        'context': 168,        # 上下文窗口
    },
    'Sentiment_v22': {
        'min_samples': 3000,   # 最少 3000 条文本
        'recommended': 10000,  # 推荐 10000+ 条
        'classes': 3,          # negative/neutral/positive
    },
}


# =============================================================================
# OHLCV 数据收集 - Kraken API
# =============================================================================

class KrakenOHLCV:
    """Kraken 官方 API 获取 OHLCV 数据"""
    
    API_URL = "https://api.kraken.com/0/public/OHLC"
    
    PAIRS = {
        'BTC': 'XXBTZUSD',
        'ETH': 'XETHZUSD',
        'SOL': 'SOLUSD',
    }
    
    def __init__(self, output_dir: str = './training_data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch(self, asset: str, interval: int = 60, since: int = None) -> pd.DataFrame:
        """
        获取 OHLCV 数据
        
        Args:
            asset: BTC, ETH, SOL
            interval: 分钟 (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Unix timestamp
        """
        if asset not in self.PAIRS:
            raise ValueError(f"不支持: {asset}, 可用: {list(self.PAIRS.keys())}")
        
        pair = self.PAIRS[asset]
        
        params = {
            'pair': pair,
            'interval': interval,
        }
        if since:
            params['since'] = since
        
        logger.info(f"获取 {asset} 数据 (Kraken API)...")
        
        try:
            response = requests.get(self.API_URL, params=params, timeout=30)
            data = response.json()
            
            if data.get('error'):
                logger.error(f"API 错误: {data['error']}")
                return None
            
            # 解析数据
            result_key = list(data['result'].keys())[0]
            if result_key == 'last':
                result_key = list(data['result'].keys())[1]
            
            ohlcv = data['result'][result_key]
            
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"ON {asset}: {len(df)} 条数据")
            logger.info(f"   范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"获取失败: {e}")
            return None
    
    def fetch_full_history(self, asset: str, days: int = 730) -> pd.DataFrame:
        """获取完整历史数据 (分批获取)"""
        all_data = []
        
        # Kraken API 每次返回 720 条
        # 需要多次调用
        
        since = int((datetime.now() - timedelta(days=days)).timestamp())
        
        while True:
            df = self.fetch(asset, interval=60, since=since)
            if df is None or len(df) == 0:
                break
            
            all_data.append(df)
            
            # 更新 since 为最后一条的时间戳
            last_ts = df['timestamp'].max()
            since = int(last_ts.timestamp())
            
            if len(df) < 720:  # 没有更多数据
                break
            
            time.sleep(1)  # 避免 rate limit
        
        if not all_data:
            return None
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        
        return result
    
    def download_all(self, assets: List[str] = ['BTC', 'ETH', 'SOL'], days: int = 730):
        """下载所有资产"""
        results = {}
        
        for asset in assets:
            logger.info(f"\n{'='*50}")
            logger.info(f"下载 {asset}...")
            logger.info('='*50)
            
            df = self.fetch_full_history(asset, days)
            
            if df is not None and len(df) > 0:
                # 保存
                output_path = self.output_dir / f"{asset}_60m.parquet"
                df.to_parquet(output_path)
                logger.info(f"ON 保存: {output_path} ({len(df)} 条)")
                results[asset] = df
            else:
                logger.error(f"OFF {asset} 下载失败")
        
        return results


# =============================================================================
# OHLCV 数据收集 - Binance Data Vision (推荐大量数据)
# =============================================================================

class BinanceDataVision:
    """
    Binance Data Vision - 免费完整历史数据
    
    数据下载页面: https://data.binance.vision/
    """
    
    BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
    
    SYMBOLS = {
        'BTC': 'BTCUSDT',
        'ETH': 'ETHUSDT',
        'SOL': 'SOLUSDT',
    }
    
    def __init__(self, output_dir: str = './training_data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_month(self, asset: str, year: int, month: int, interval: str = '1h') -> pd.DataFrame:
        """下载单月数据"""
        symbol = self.SYMBOLS.get(asset)
        if not symbol:
            raise ValueError(f"不支持: {asset}")
        
        # URL 格式: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip
        month_str = f"{year}-{month:02d}"
        filename = f"{symbol}-{interval}-{month_str}.zip"
        url = f"{self.BASE_URL}/{symbol}/{interval}/{filename}"
        
        try:
            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                return None
            
            # 解压 ZIP
            import zipfile
            from io import BytesIO
            
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None)
            
            # Binance klines 格式
            df.columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logger.debug(f"下载失败 {month_str}: {e}")
            return None
    
    def download_range(self, asset: str, start_year: int, start_month: int, 
                       end_year: int = None, end_month: int = None) -> pd.DataFrame:
        """下载日期范围内的数据"""
        if end_year is None:
            end_year = datetime.now().year
        if end_month is None:
            end_month = datetime.now().month
        
        all_data = []
        
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)
        
        logger.info(f"下载 {asset} 从 {start_year}-{start_month:02d} 到 {end_year}-{end_month:02d}...")
        
        while current <= end:
            df = self.download_month(asset, current.year, current.month)
            if df is not None:
                all_data.append(df)
                logger.info(f"  ✓ {current.year}-{current.month:02d}: {len(df)} 条")
            
            # 下个月
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
            
            time.sleep(0.5)
        
        if not all_data:
            return None
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"ON {asset}: 总计 {len(result)} 条")
        
        return result
    
    def download_all(self, assets: List[str] = ['BTC', 'ETH', 'SOL'], years: int = 2):
        """下载所有资产的历史数据"""
        start_year = datetime.now().year - years
        start_month = datetime.now().month
        
        results = {}
        
        for asset in assets:
            logger.info(f"\n{'='*50}")
            logger.info(f"下载 {asset} (Binance Data Vision)...")
            logger.info('='*50)
            
            df = self.download_range(asset, start_year, start_month)
            
            if df is not None and len(df) > 0:
                output_path = self.output_dir / f"{asset}_60m.parquet"
                df.to_parquet(output_path)
                logger.info(f"ON 保存: {output_path}")
                results[asset] = df
        
        return results


# =============================================================================
# 情感数据收集
# =============================================================================

class SentimentDataCollector:
    """
    情感数据收集器
    
    数据源:
    1. Kaggle 数据集 (推荐)
    2. CryptoCompare News API
    3. 手动标注数据
    """
    
    # Kaggle 推荐数据集
    KAGGLE_DATASETS = {
        'crypto_news': 'oliviervha/crypto-news-headlines-sentiment',
        'bitcoin_tweets': 'kaushiksuresh147/bitcoin-tweets',
        'crypto_sentiment': 'zespsolutions/cryptocurrencies-sentiment-analysis',
        'financial_news': 'ankurzing/sentiment-analysis-for-financial-news',
    }
    
    # CryptoCompare News API (免费)
    CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
    
    def __init__(self, output_dir: str = './training_data/sentiment'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_cryptocompare_news(self, categories: str = 'BTC,ETH,SOL', limit: int = 100) -> pd.DataFrame:
        """从 CryptoCompare 获取新闻"""
        params = {
            'categories': categories,
            'lang': 'EN',
        }
        
        all_news = []
        
        try:
            response = requests.get(self.CRYPTOCOMPARE_NEWS_URL, params=params, timeout=30)
            data = response.json()
            
            if data.get('Data'):
                for item in data['Data'][:limit]:
                    all_news.append({
                        'timestamp': datetime.fromtimestamp(item['published_on']),
                        'title': item.get('title', ''),
                        'body': item.get('body', ''),
                        'source': item.get('source', ''),
                        'categories': item.get('categories', ''),
                        'url': item.get('url', ''),
                    })
            
            df = pd.DataFrame(all_news)
            logger.info(f"ON 获取 {len(df)} 条新闻")
            return df
            
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return None
    
    def create_sample_sentiment_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        创建示例情感数据 (用于测试)
        
        实际使用时应该用真实数据
        """
        logger.info(f"生成 {n_samples} 条示例情感数据...")
        
        # 示例模板
        positive_templates = [
            "Bitcoin surges to new all-time high",
            "{coin} breaks resistance, bullish momentum continues",
            "Major institution announces {coin} investment",
            "Ethereum ETF approved, market rallies",
            "SOL ecosystem growth accelerates",
            "{coin} adoption increases among retail investors",
            "Positive regulatory news boosts crypto market",
        ]
        
        negative_templates = [
            "Bitcoin crashes below key support level",
            "{coin} plunges amid market sell-off",
            "Major exchange hack causes panic selling",
            "Regulatory crackdown threatens {coin}",
            "SOL network experiences outage",
            "{coin} faces liquidation cascade",
            "Bear market fears intensify",
        ]
        
        neutral_templates = [
            "{coin} trades sideways in consolidation phase",
            "Market awaits Federal Reserve decision",
            "Mixed signals for {coin} price action",
            "Trading volume remains average",
            "Analysts divided on {coin} outlook",
        ]
        
        coins = ['Bitcoin', 'BTC', 'Ethereum', 'ETH', 'Solana', 'SOL']
        
        data = []
        for i in range(n_samples):
            coin = np.random.choice(coins)
            
            # 40% positive, 30% negative, 30% neutral
            r = np.random.random()
            if r < 0.4:
                template = np.random.choice(positive_templates)
                label = 2  # positive
            elif r < 0.7:
                template = np.random.choice(negative_templates)
                label = 0  # negative
            else:
                template = np.random.choice(neutral_templates)
                label = 1  # neutral
            
            text = template.format(coin=coin)
            
            data.append({
                'text': text,
                'label': label,
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 8760)),
            })
        
        df = pd.DataFrame(data)
        logger.info(f"ON 生成完成: positive={len(df[df.label==2])}, neutral={len(df[df.label==1])}, negative={len(df[df.label==0])}")
        
        return df
    
    def download_kaggle_dataset(self, dataset_name: str) -> bool:
        """
        下载 Kaggle 数据集
        
        需要先配置 Kaggle API:
        1. 创建 Kaggle 账号
        2. 在 Account 页面创建 API Token
        3. 将 kaggle.json 放到 ~/.kaggle/
        """
        try:
            import kaggle
            
            logger.info(f"下载 Kaggle 数据集: {dataset_name}")
            kaggle.api.dataset_download_files(dataset_name, path=str(self.output_dir), unzip=True)
            logger.info(f"ON 下载完成: {self.output_dir}")
            return True
            
        except ImportError:
            logger.error("请先安装 kaggle: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"下载失败: {e}")
            logger.info("请确保已配置 Kaggle API (~/.kaggle/kaggle.json)")
            return False


# =============================================================================
# 数据验证
# =============================================================================

def validate_data(data_dir: str = './training_data'):
    """验证数据是否满足训练需求"""
    data_dir = Path(data_dir)
    
    print("\n" + "="*60)
    print("数据验证")
    print("="*60)
    
    # 检查 OHLCV 数据
    print("\n--- OHLCV 数据 (DT v3.2, DRL v5.5) ---")
    ohlcv_dir = data_dir / 'raw'
    
    total_rows = 0
    for asset in ['BTC', 'ETH', 'SOL']:
        file_path = ohlcv_dir / f"{asset}_60m.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            rows = len(df)
            total_rows += rows
            
            dt_status = "ON" if rows >= DATA_REQUIREMENTS['DT_v32']['min_rows'] else "OFF"
            drl_status = "ON" if rows >= DATA_REQUIREMENTS['DRL_v55']['min_rows'] else "OFF"
            
            print(f"  {asset}: {rows:,} 条 ({rows/24:.0f} 天)")
            print(f"    DT v3.2:  {dt_status} (需要 {DATA_REQUIREMENTS['DT_v32']['min_rows']:,}+)")
            print(f"    DRL v5.5: {drl_status} (需要 {DATA_REQUIREMENTS['DRL_v55']['min_rows']:,}+)")
        else:
            print(f"  {asset}: OFF 文件不存在")
    
    # 检查情感数据
    print("\n--- 情感数据 (Sentiment v2.2) ---")
    sentiment_dir = data_dir / 'sentiment'
    
    sentiment_files = list(sentiment_dir.glob('*.csv')) + list(sentiment_dir.glob('*.parquet'))
    if sentiment_files:
        for f in sentiment_files:
            if f.suffix == '.csv':
                df = pd.read_csv(f)
            else:
                df = pd.read_parquet(f)
            
            status = "ON" if len(df) >= DATA_REQUIREMENTS['Sentiment_v22']['min_samples'] else "OFF"
            print(f"  {f.name}: {len(df):,} 条 {status} (需要 {DATA_REQUIREMENTS['Sentiment_v22']['min_samples']:,}+)")
    else:
        print("  OFF 无情感数据文件")
    
    print("\n" + "="*60)
    
    return total_rows


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMATS 数据获取')
    
    # 数据类型
    parser.add_argument('--ohlcv', action='store_true', help='下载 OHLCV 价格数据')
    parser.add_argument('--sentiment', action='store_true', help='下载/生成情感数据')
    parser.add_argument('--all', action='store_true', help='下载全部数据')
    
    # 数据源
    parser.add_argument('--source', type=str, default='binance',
                       choices=['kraken', 'binance', 'cryptocompare'],
                       help='数据源 (默认: binance)')
    
    # 参数
    parser.add_argument('--assets', type=str, default='BTC,ETH,SOL', help='资产列表')
    parser.add_argument('--years', type=int, default=2, help='历史数据年数')
    parser.add_argument('--output', type=str, default='./training_data', help='输出目录')
    
    # 导入本地数据
    parser.add_argument('--import-csv', type=str, help='导入本地 CSV 文件')
    parser.add_argument('--asset', type=str, help='导入数据的资产名称')
    
    # 验证
    parser.add_argument('--validate', action='store_true', help='验证数据')
    
    # 生成示例数据
    parser.add_argument('--generate-sample', action='store_true', help='生成示例数据 (用于测试)')
    
    args = parser.parse_args()
    
    # 验证
    if args.validate:
        validate_data(args.output)
        return
    
    # 导入本地数据
    if args.import_csv:
        logger.info(f"导入本地数据: {args.import_csv}")
        df = pd.read_csv(args.import_csv)
        
        # 标准化
        column_map = {
            'Date': 'timestamp', 'date': 'timestamp', 'time': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume',
        }
        df = df.rename(columns=column_map)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 保存
        asset = args.asset or 'IMPORTED'
        output_dir = Path(args.output) / 'raw'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{asset}_60m.parquet"
        df.to_parquet(output_path)
        logger.info(f"ON 保存: {output_path} ({len(df)} 条)")
        return
    
    # 生成示例数据
    if args.generate_sample:
        logger.info("生成示例数据...")
        
        # OHLCV
        ohlcv_dir = Path(args.output) / 'raw'
        ohlcv_dir.mkdir(parents=True, exist_ok=True)
        
        for asset in args.assets.split(','):
            n_rows = 20000
            dates = pd.date_range(end=datetime.now(), periods=n_rows, freq='H')
            
            price = 100
            prices = [price]
            for _ in range(n_rows - 1):
                price *= (1 + np.random.normal(0, 0.01))
                prices.append(price)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'volume': np.random.uniform(1000, 10000, n_rows),
            })
            
            output_path = ohlcv_dir / f"{asset.strip()}_60m.parquet"
            df.to_parquet(output_path)
            logger.info(f"ON 生成: {output_path} ({len(df)} 条)")
        
        # 情感数据
        sentiment_collector = SentimentDataCollector(f"{args.output}/sentiment")
        df = sentiment_collector.create_sample_sentiment_data(10000)
        output_path = Path(args.output) / 'sentiment' / 'sample_sentiment.parquet'
        df.to_parquet(output_path)
        logger.info(f"ON 生成: {output_path}")
        
        validate_data(args.output)
        return
    
    # 下载数据
    assets = [a.strip() for a in args.assets.split(',')]
    
    if args.all or args.ohlcv:
        logger.info(f"\n下载 OHLCV 数据 (源: {args.source})...")
        
        if args.source == 'binance':
            collector = BinanceDataVision(f"{args.output}/raw")
            collector.download_all(assets, years=args.years)
        else:
            collector = KrakenOHLCV(f"{args.output}/raw")
            collector.download_all(assets)
    
    if args.all or args.sentiment:
        logger.info("\n下载/生成情感数据...")
        sentiment_collector = SentimentDataCollector(f"{args.output}/sentiment")
        
        # 尝试获取新闻
        df = sentiment_collector.fetch_cryptocompare_news(limit=1000)
        if df is not None:
            df.to_parquet(f"{args.output}/sentiment/crypto_news.parquet")
        
        # 生成示例数据补充
        sample_df = sentiment_collector.create_sample_sentiment_data(5000)
        sample_df.to_parquet(f"{args.output}/sentiment/sample_sentiment.parquet")
    
    # 验证结果
    validate_data(args.output)


if __name__ == "__main__":
    main()

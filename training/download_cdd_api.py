#!/usr/bin/env python3
"""
================================================================================
CryptoDataDownload API 数据下载脚本
================================================================================

用于 HMATS 训练的完整数据下载方案。

用法:
    # 设置 API Token
    set CDD_API_TOKEN=your_token_here  (Windows)
    export CDD_API_TOKEN=your_token_here  (Linux/Mac)
    
    # 运行下载
    python download_cdd_api.py --years 5 --assets SOL BTC ETH
    
    # 或指定 token
    python download_cdd_api.py --token YOUR_TOKEN --years 5

数据来源: https://www.cryptodatadownload.com/
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CDD-Download")

# =============================================================================
# 配置
# =============================================================================

API_BASE = "https://api.cryptodatadownload.com/v1"

# 资产配置
ASSETS = {
    'SOL': {
        'spot_symbol': 'SOLUSDT',
        'futures_symbol': 'SOLUSDT',
        'launch_date': '2020-04-10',  # SOL 上线日期
    },
    'BTC': {
        'spot_symbol': 'BTCUSDT',
        'futures_symbol': 'BTCUSDT',
        'launch_date': '2017-08-17',
    },
    'ETH': {
        'spot_symbol': 'ETHUSDT',
        'futures_symbol': 'ETHUSDT',
        'launch_date': '2017-08-17',
    },
}

# 数据端点配置
ENDPOINTS = {
    'spot_ohlcv': {
        'path': '/ohlc/{exchange}/spot/{symbol}',
        'description': '现货 OHLCV',
        'required': True,
    },
    'futures_ohlcv': {
        'path': '/ohlc/{exchange}/futures/{symbol}',
        'description': '期货 OHLCV',
        'required': True,
    },
    'funding_rate': {
        'path': '/data/{exchange}/futures/{symbol}/funding',
        'description': 'Funding Rate',
        'required': True,
    },
    'open_interest': {
        'path': '/data/{exchange}/futures/{symbol}/oi',
        'description': 'Open Interest',
        'required': False,
    },
}


# =============================================================================
# API 客户端
# =============================================================================

class CryptoDataDownloadClient:
    """CryptoDataDownload API 客户端"""
    
    def __init__(self, api_token: str, rate_limit_delay: float = 1.0):
        self.api_token = api_token
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_token}',
            'Accept': 'application/json',
        })
    
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送 API 请求"""
        url = API_BASE + endpoint
        
        try:
            logger.debug(f"请求: {url}")
            logger.debug(f"参数: {params}")
            
            resp = self.session.get(url, params=params, timeout=60)
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 401:
                logger.error("OFF API Token 无效或已过期")
                return None
            elif resp.status_code == 429:
                logger.warning("[WARN]️ Rate limit，等待 60 秒...")
                time.sleep(60)
                return self._request(endpoint, params)  # 重试
            else:
                logger.error(f"OFF API 错误 {resp.status_code}: {resp.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("OFF 请求超时")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"OFF 网络错误: {e}")
            return None
        finally:
            time.sleep(self.rate_limit_delay)  # Rate limit
    
    def get_ohlcv(self, exchange: str, market_type: str, symbol: str,
                  timeframe: str = '1h', start_date: str = None, 
                  end_date: str = None) -> Optional[pd.DataFrame]:
        """获取 OHLCV 数据"""
        
        # 构建端点
        endpoint = f"/ohlc/{exchange.lower()}/{market_type.lower()}/{symbol.upper()}"
        
        params = {
            'timeframe': timeframe,
        }
        if start_date:
            params['start'] = start_date
        if end_date:
            params['end'] = end_date
        
        logger.info(f"📥 下载 {exchange} {market_type} {symbol} ({timeframe})...")
        
        data = self._request(endpoint, params)
        
        if data is None:
            return None
        
        # 解析数据
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            logger.warning(f"[WARN]️ 未知数据格式: {type(data)}")
            df = pd.DataFrame(data) if data else None
        
        if df is not None and len(df) > 0:
            # 标准化列名
            df.columns = df.columns.str.lower()
            
            # 确保有必要的列
            col_mapping = {
                'date': 'timestamp',
                'time': 'timestamp',
                'datetime': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vol': 'volume',
            }
            df = df.rename(columns=col_mapping)
            
            # 转换时间戳
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"  ON {len(df)} 行数据")
        
        return df
    
    def get_funding_rate(self, exchange: str, symbol: str,
                         start_date: str = None) -> Optional[pd.DataFrame]:
        """获取 Funding Rate 数据"""
        
        endpoint = f"/data/{exchange.lower()}/futures/{symbol.upper()}/funding"
        
        params = {}
        if start_date:
            params['start'] = start_date
        
        logger.info(f"📥 下载 {symbol} Funding Rate...")
        
        data = self._request(endpoint, params)
        
        if data is None:
            return None
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data) if data else None
        
        if df is not None and len(df) > 0:
            df.columns = df.columns.str.lower()
            logger.info(f"  ON {len(df)} 行数据")
        
        return df
    
    def get_open_interest(self, exchange: str, symbol: str,
                          start_date: str = None) -> Optional[pd.DataFrame]:
        """获取 Open Interest 数据"""
        
        endpoint = f"/data/{exchange.lower()}/futures/{symbol.upper()}/oi"
        
        params = {}
        if start_date:
            params['start'] = start_date
        
        logger.info(f"📥 下载 {symbol} Open Interest...")
        
        data = self._request(endpoint, params)
        
        if data is None:
            return None
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data) if data else None
        
        if df is not None and len(df) > 0:
            df.columns = df.columns.str.lower()
            logger.info(f"  ON {len(df)} 行数据")
        
        return df
    
    def get_volatility(self, exchange: str = 'deribit', 
                       symbol: str = 'BTC') -> Optional[pd.DataFrame]:
        """获取波动率数据 (Deribit)"""
        
        endpoint = f"/data/{exchange.lower()}/volatility/{symbol.upper()}"
        
        logger.info(f"📥 下载 {symbol} 波动率...")
        
        data = self._request(endpoint)
        
        if data is None:
            return None
        
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('data', []))
        
        if df is not None and len(df) > 0:
            df.columns = df.columns.str.lower()
            logger.info(f"  ON {len(df)} 行数据")
        
        return df


# =============================================================================
# 下载管理器
# =============================================================================

class DataDownloader:
    """数据下载管理器"""
    
    def __init__(self, client: CryptoDataDownloadClient, output_dir: str = 'training_data'):
        self.client = client
        self.output_dir = Path(output_dir)
        
        # 创建目录
        (self.output_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'futures').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'funding').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    
    def download_asset(self, asset: str, years: int = 5, 
                       exchange: str = 'binance') -> Dict[str, pd.DataFrame]:
        """下载单个资产的所有数据"""
        
        if asset not in ASSETS:
            logger.error(f"OFF 未知资产: {asset}")
            return {}
        
        config = ASSETS[asset]
        results = {}
        
        # 计算日期范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
        
        # 检查资产上线日期
        launch_date = config['launch_date']
        if start_date < launch_date:
            logger.warning(f"[WARN]️ {asset} 上线于 {launch_date}，调整开始日期")
            start_date = launch_date
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📦 下载 {asset} 数据 ({start_date} ~ {end_date})")
        logger.info(f"{'='*60}")
        
        # 1. 现货 OHLCV
        logger.info(f"\n[1/4] 现货 OHLCV")
        df_spot = self.client.get_ohlcv(
            exchange, 'spot', config['spot_symbol'],
            timeframe='1h', start_date=start_date, end_date=end_date
        )
        if df_spot is not None and len(df_spot) > 0:
            path = self.output_dir / 'raw' / f'{asset}_60m.parquet'
            df_spot.to_parquet(path)
            results['spot'] = df_spot
            logger.info(f"  💾 保存: {path}")
        
        # 2. 期货 OHLCV
        logger.info(f"\n[2/4] 期货 OHLCV")
        df_futures = self.client.get_ohlcv(
            exchange, 'futures', config['futures_symbol'],
            timeframe='1h', start_date=start_date, end_date=end_date
        )
        if df_futures is not None and len(df_futures) > 0:
            path = self.output_dir / 'futures' / f'{asset}_futures_60m.parquet'
            df_futures.to_parquet(path)
            results['futures'] = df_futures
            logger.info(f"  💾 保存: {path}")
        
        # 3. Funding Rate
        logger.info(f"\n[3/4] Funding Rate")
        df_funding = self.client.get_funding_rate(
            exchange, config['futures_symbol'], start_date=start_date
        )
        if df_funding is not None and len(df_funding) > 0:
            path = self.output_dir / 'funding' / f'{asset}_funding.parquet'
            df_funding.to_parquet(path)
            results['funding'] = df_funding
            logger.info(f"  💾 保存: {path}")
        
        # 4. Open Interest (可选)
        logger.info(f"\n[4/4] Open Interest")
        df_oi = self.client.get_open_interest(
            exchange, config['futures_symbol'], start_date=start_date
        )
        if df_oi is not None and len(df_oi) > 0:
            path = self.output_dir / 'metrics' / f'{asset}_oi.parquet'
            df_oi.to_parquet(path)
            results['oi'] = df_oi
            logger.info(f"  💾 保存: {path}")
        
        return results
    
    def download_all(self, assets: List[str], years: int = 5,
                     exchange: str = 'binance') -> Dict[str, Dict]:
        """下载所有资产数据"""
        
        all_results = {}
        
        for asset in assets:
            try:
                results = self.download_asset(asset, years, exchange)
                all_results[asset] = results
            except Exception as e:
                logger.error(f"OFF 下载 {asset} 失败: {e}")
                continue
        
        # 打印摘要
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Dict]):
        """打印下载摘要"""
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 下载摘要")
        logger.info(f"{'='*60}")
        
        total_rows = 0
        
        for asset, data in results.items():
            logger.info(f"\n{asset}:")
            for data_type, df in data.items():
                if df is not None:
                    rows = len(df)
                    total_rows += rows
                    
                    # 计算日期范围
                    if 'timestamp' in df.columns:
                        start = df['timestamp'].min()
                        end = df['timestamp'].max()
                        days = (end - start).days
                        logger.info(f"  {data_type}: {rows:,} 行 ({days} 天)")
                    else:
                        logger.info(f"  {data_type}: {rows:,} 行")
        
        logger.info(f"\n总计: {total_rows:,} 行数据")
        logger.info(f"{'='*60}")


# =============================================================================
# 备用方案: 免费 CSV 下载
# =============================================================================

def download_free_csv(asset: str, output_dir: str = 'training_data'):
    """
    从 CryptoDataDownload 免费 CSV 下载
    (无需 API Token，但数据可能不完整)
    """
    
    output_path = Path(output_dir) / 'raw'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 免费 CSV 下载链接
    urls = {
        'BTC': 'https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv',
        'ETH': 'https://www.cryptodatadownload.com/cdd/Binance_ETHUSDT_1h.csv',
        'SOL': 'https://www.cryptodatadownload.com/cdd/Binance_SOLUSDT_1h.csv',
    }
    
    if asset not in urls:
        logger.error(f"OFF 无免费 CSV: {asset}")
        return None
    
    url = urls[asset]
    logger.info(f"📥 下载免费 CSV: {asset}...")
    
    try:
        # 跳过第一行 (网站说明)
        df = pd.read_csv(url, skiprows=1)
        
        # 标准化列名
        df.columns = df.columns.str.lower().str.strip()
        
        col_mapping = {
            'date': 'timestamp',
            'unix': 'unix_timestamp',
            'symbol': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'volume usdt': 'volume_usdt',
            'tradecount': 'trade_count',
        }
        df = df.rename(columns=col_mapping)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 保存
        path = output_path / f'{asset}_60m.parquet'
        df.to_parquet(path)
        
        logger.info(f"  ON {len(df)} 行, 保存到 {path}")
        
        return df
        
    except Exception as e:
        logger.error(f"OFF 下载失败: {e}")
        return None


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CryptoDataDownload API 数据下载',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用 API Token 下载 5 年数据
    python download_cdd_api.py --token YOUR_TOKEN --years 5
    
    # 使用环境变量
    set CDD_API_TOKEN=your_token
    python download_cdd_api.py --years 5
    
    # 下载免费 CSV (无需 Token)
    python download_cdd_api.py --free
        """
    )
    
    parser.add_argument('--token', type=str, default=None,
                        help='CryptoDataDownload API Token')
    parser.add_argument('--years', type=int, default=5,
                        help='下载年数 (默认 5)')
    parser.add_argument('--assets', nargs='+', default=['SOL', 'BTC', 'ETH'],
                        help='资产列表 (默认 SOL BTC ETH)')
    parser.add_argument('--exchange', type=str, default='binance',
                        help='交易所 (默认 binance)')
    parser.add_argument('--output', type=str, default='training_data',
                        help='输出目录 (默认 training_data)')
    parser.add_argument('--free', action='store_true',
                        help='使用免费 CSV 下载 (无需 Token)')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                        help='请求间隔秒数 (默认 1.0)')
    
    args = parser.parse_args()
    
    # 免费 CSV 模式
    if args.free:
        logger.info("📥 使用免费 CSV 下载模式")
        logger.warning("[WARN]️ 免费 CSV 数据可能不完整，建议使用 API")
        
        for asset in args.assets:
            download_free_csv(asset, args.output)
        
        logger.info("\nON 下载完成!")
        return
    
    # API 模式
    api_token = args.token or os.environ.get('CDD_API_TOKEN')
    
    if not api_token:
        logger.error("OFF 未提供 API Token!")
        logger.error("   使用 --token YOUR_TOKEN 或设置环境变量 CDD_API_TOKEN")
        logger.error("   或使用 --free 下载免费 CSV")
        sys.exit(1)
    
    logger.info(f"🔑 API Token: {api_token[:8]}...")
    logger.info(f"📅 下载年数: {args.years}")
    logger.info(f"📦 资产: {args.assets}")
    logger.info(f"🏦 交易所: {args.exchange}")
    
    # 创建客户端
    client = CryptoDataDownloadClient(api_token, args.rate_limit)
    downloader = DataDownloader(client, args.output)
    
    # 下载数据
    results = downloader.download_all(args.assets, args.years, args.exchange)
    
    logger.info("\nON 下载完成!")
    
    # 检查数据量是否足够
    total_rows = sum(
        len(df) for asset_data in results.values() 
        for df in asset_data.values() if df is not None
    )
    
    if total_rows < 10000:
        logger.warning("[WARN]️ 数据量较少，建议检查 API 访问权限")
    
    # 提示下一步
    logger.info("\n📋 下一步:")
    logger.info("   python check_data.py  # 验证数据")
    logger.info("   python run_training.py --epochs 100  # 开始训练")


if __name__ == "__main__":
    main()

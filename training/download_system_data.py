#!/usr/bin/env python3
"""
================================================================================
HMATS 系统运行数据下载脚本
================================================================================

下载系统运行时需要的实时/周期性数据。

用法:
    # 一次性下载最新数据
    python download_system_data.py --token YOUR_TOKEN
    
    # 设置定时任务 (Windows Task Scheduler / Linux cron)
    # 每小时运行一次

数据来源: CryptoDataDownload Plus+ API
================================================================================
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("System-Data")

# =============================================================================
# 配置
# =============================================================================

API_BASE = "https://api.cryptodatadownload.com/v1"

ASSETS = ['SOL', 'BTC', 'ETH']
SYMBOLS = {
    'SOL': 'SOLUSDT',
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
}


# =============================================================================
# API 客户端
# =============================================================================

class CDDClient:
    """CryptoDataDownload API 客户端"""
    
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
        })
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """发送请求"""
        url = API_BASE + endpoint
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(10)
                return self._request(endpoint, params)
            else:
                logger.warning(f"[WARN]️ {resp.status_code}: {endpoint}")
                return None
                
        except Exception as e:
            logger.error(f"OFF 请求失败: {e}")
            return None
    
    def get_latest_ohlcv(self, symbol: str, hours: int = 168) -> Optional[pd.DataFrame]:
        """获取最近 N 小时 OHLCV"""
        
        start = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        
        endpoint = f"/ohlc/binance/spot/{symbol}"
        params = {'timeframe': '1h', 'start': start, 'end': end}
        
        data = self._request(endpoint, params)
        
        if data is None:
            return None
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            return None
        
        if len(df) > 0:
            df.columns = df.columns.str.lower()
            col_map = {'date': 'timestamp', 'time': 'timestamp', 't': 'timestamp',
                      'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
            df = df.rename(columns=col_map)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
        
        return df
    
    def get_latest_funding(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """获取最近 Funding Rate"""
        
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        endpoints = [
            f"/data/binance/futures/{symbol}/funding",
            f"/summary/binance/futures/funding/{symbol}",
        ]
        
        for endpoint in endpoints:
            data = self._request(endpoint, {'start': start})
            if data:
                df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('data', []))
                if len(df) > 0:
                    df.columns = df.columns.str.lower()
                    return df
        
        return None
    
    def get_latest_oi(self, symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
        """获取最近 Open Interest"""
        
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        endpoints = [
            f"/data/binance/futures/{symbol}/oi",
            f"/summary/binance/futures/metrics/{symbol}",
        ]
        
        for endpoint in endpoints:
            data = self._request(endpoint, {'start': start})
            if data:
                df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('data', []))
                if len(df) > 0:
                    df.columns = df.columns.str.lower()
                    return df
        
        return None
    
    def get_correlations(self) -> Optional[pd.DataFrame]:
        """获取资产相关性"""
        
        endpoint = "/risk/correlations"
        data = self._request(endpoint)
        
        if data:
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('data', []))
            if len(df) > 0:
                df.columns = df.columns.str.lower()
                return df
        
        return None
    
    def get_volatility_index(self, symbol: str = 'BTC') -> Optional[pd.DataFrame]:
        """获取波动率指数"""
        
        endpoint = f"/summary/deribit/volatility/{symbol}"
        data = self._request(endpoint)
        
        if data:
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data.get('data', []))
            if len(df) > 0:
                df.columns = df.columns.str.lower()
                return df
        
        return None


# =============================================================================
# 系统数据下载器
# =============================================================================

class SystemDataDownloader:
    """系统运行数据下载器"""
    
    def __init__(self, client: CDDClient, output_dir: str = 'system_data'):
        self.client = client
        self.output_dir = Path(output_dir)
        
        # 创建目录
        (self.output_dir / 'ohlcv').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'funding').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'oi').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'risk').mkdir(parents=True, exist_ok=True)
    
    def update_ohlcv(self, hours: int = 168) -> Dict[str, int]:
        """更新 OHLCV 数据 (最近 7 天)"""
        
        logger.info(f"\n📥 更新 OHLCV (最近 {hours} 小时)...")
        
        results = {}
        
        for asset in ASSETS:
            symbol = SYMBOLS[asset]
            df = self.client.get_latest_ohlcv(symbol, hours)
            
            if df is not None and len(df) > 0:
                path = self.output_dir / 'ohlcv' / f'{asset}_latest.parquet'
                df.to_parquet(path)
                results[asset] = len(df)
                logger.info(f"  ON {asset}: {len(df)} 行")
            else:
                logger.warning(f"  [WARN]️ {asset}: 无数据")
        
        return results
    
    def update_funding(self, days: int = 30) -> Dict[str, int]:
        """更新 Funding Rate (最近 30 天)"""
        
        logger.info(f"\n📥 更新 Funding Rate (最近 {days} 天)...")
        
        results = {}
        
        for asset in ASSETS:
            symbol = SYMBOLS[asset]
            df = self.client.get_latest_funding(symbol, days)
            
            if df is not None and len(df) > 0:
                path = self.output_dir / 'funding' / f'{asset}_funding_latest.parquet'
                df.to_parquet(path)
                results[asset] = len(df)
                logger.info(f"  ON {asset}: {len(df)} 行")
            else:
                logger.warning(f"  [WARN]️ {asset}: 无数据")
        
        return results
    
    def update_open_interest(self, days: int = 7) -> Dict[str, int]:
        """更新 Open Interest (最近 7 天)"""
        
        logger.info(f"\n📥 更新 Open Interest (最近 {days} 天)...")
        
        results = {}
        
        for asset in ASSETS:
            symbol = SYMBOLS[asset]
            df = self.client.get_latest_oi(symbol, days)
            
            if df is not None and len(df) > 0:
                path = self.output_dir / 'oi' / f'{asset}_oi_latest.parquet'
                df.to_parquet(path)
                results[asset] = len(df)
                logger.info(f"  ON {asset}: {len(df)} 行")
            else:
                logger.warning(f"  [WARN]️ {asset}: 无数据")
        
        return results
    
    def update_risk_metrics(self) -> Dict[str, int]:
        """更新风险指标"""
        
        logger.info(f"\n📥 更新风险指标...")
        
        results = {}
        
        # 相关性矩阵
        df = self.client.get_correlations()
        if df is not None and len(df) > 0:
            path = self.output_dir / 'risk' / 'correlations.parquet'
            df.to_parquet(path)
            results['correlations'] = len(df)
            logger.info(f"  ON 相关性: {len(df)} 行")
        
        # 波动率指数
        for symbol in ['BTC', 'ETH']:
            df = self.client.get_volatility_index(symbol)
            if df is not None and len(df) > 0:
                path = self.output_dir / 'risk' / f'{symbol}_volatility.parquet'
                df.to_parquet(path)
                results[f'{symbol}_vol'] = len(df)
                logger.info(f"  ON {symbol} 波动率: {len(df)} 行")
        
        return results
    
    def update_all(self):
        """更新所有系统数据"""
        
        logger.info("=" * 60)
        logger.info(f"🔄 HMATS 系统数据更新")
        logger.info(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        results = {}
        
        results['ohlcv'] = self.update_ohlcv()
        results['funding'] = self.update_funding()
        results['oi'] = self.update_open_interest()
        results['risk'] = self.update_risk_metrics()
        
        # 摘要
        total = sum(
            sum(v.values()) if isinstance(v, dict) else 0 
            for v in results.values()
        )
        
        logger.info("\n" + "=" * 60)
        logger.info(f"📊 更新完成: {total:,} 行数据")
        logger.info(f"📁 数据目录: {self.output_dir}/")
        logger.info("=" * 60)
        
        # 保存更新时间戳
        timestamp_file = self.output_dir / 'last_update.txt'
        with open(timestamp_file, 'w', encoding="utf-8") as f:
            f.write(datetime.now().isoformat())
        
        return results


# =============================================================================
# 生成系统输入数据
# =============================================================================

def prepare_system_input(data_dir: str = 'system_data', 
                         output_file: str = 'system_data/system_input.parquet'):
    """
    整合所有系统数据为单一输入文件
    供 HMATS 实时决策使用
    """
    
    data_dir = Path(data_dir)
    
    logger.info("\n📦 生成系统输入数据...")
    
    dfs = {}
    
    # 加载 OHLCV
    for asset in ASSETS:
        path = data_dir / 'ohlcv' / f'{asset}_latest.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['asset'] = asset
            dfs[f'{asset}_ohlcv'] = df
    
    # 加载 Funding
    for asset in ASSETS:
        path = data_dir / 'funding' / f'{asset}_funding_latest.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['asset'] = asset
            dfs[f'{asset}_funding'] = df
    
    # 加载 OI
    for asset in ASSETS:
        path = data_dir / 'oi' / f'{asset}_oi_latest.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['asset'] = asset
            dfs[f'{asset}_oi'] = df
    
    # 合并 OHLCV
    if any('ohlcv' in k for k in dfs):
        ohlcv_dfs = [v for k, v in dfs.items() if 'ohlcv' in k]
        combined = pd.concat(ohlcv_dfs, ignore_index=True)
        combined.to_parquet(output_file)
        logger.info(f"  ON 系统输入: {len(combined)} 行 -> {output_file}")
        return combined
    
    logger.warning("  [WARN]️ 无数据可合并")
    return None


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HMATS 系统数据更新')
    parser.add_argument('--token', type=str, required=True, help='API Token')
    parser.add_argument('--output', type=str, default='system_data', help='输出目录')
    parser.add_argument('--prepare', action='store_true', help='生成系统输入文件')
    
    args = parser.parse_args()
    
    client = CDDClient(args.token)
    downloader = SystemDataDownloader(client, args.output)
    
    # 更新数据
    downloader.update_all()
    
    # 生成系统输入
    if args.prepare:
        prepare_system_input(args.output)
    
    logger.info("\nON 系统数据更新完成!")


if __name__ == "__main__":
    main()

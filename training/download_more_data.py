#!/usr/bin/env python3
"""
================================================================================
批量下载 Binance 历史数据 - 获取 2-3 年完整数据
================================================================================

用法:
    python download_more_data.py --years 2
    python download_more_data.py --start 2022-01 --end 2025-01
    
预期数据量:
    1 年 ≈ 8,760 条 (365 × 24)
    2 年 ≈ 17,520 条
    3 年 ≈ 26,280 条

================================================================================
"""

import os
import sys
import argparse
import requests
import zipfile
from io import BytesIO
from pathlib import Path
from datetime import datetime
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger()

# Binance Data Vision 基础 URL
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

SYMBOLS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT', 
    'SOL': 'SOLUSDT',
}


def download_month(symbol: str, year: int, month: int, interval: str = '1h') -> pd.DataFrame:
    """下载单月数据"""
    month_str = f"{year}-{month:02d}"
    filename = f"{symbol}-{interval}-{month_str}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{filename}"
    
    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            return None
        
        # 解压 ZIP
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None)
        
        # Binance klines 格式: open_time, open, high, low, close, volume, ...
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


def download_asset(asset: str, start_year: int, start_month: int, 
                   end_year: int, end_month: int, output_dir: Path) -> int:
    """下载单个资产的全部数据"""
    
    symbol = SYMBOLS.get(asset)
    if not symbol:
        logger.error(f"不支持的资产: {asset}")
        return 0
    
    all_data = []
    
    current = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"下载 {asset} ({symbol})")
    logger.info(f"范围: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d}")
    logger.info('='*50)
    
    while current <= end:
        df = download_month(symbol, current.year, current.month)
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            logger.info(f"  ✓ {current.year}-{current.month:02d}: {len(df):,} 条")
        else:
            logger.warning(f"  ✗ {current.year}-{current.month:02d}: 无数据")
        
        # 下个月
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
        
        time.sleep(0.3)  # 避免请求过快
    
    if not all_data:
        logger.error(f"OFF {asset}: 未获取到任何数据")
        return 0
    
    # 合并数据
    result = pd.concat(all_data, ignore_index=True)
    result = result.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    
    # 保存
    output_path = output_dir / f"{asset}_60m.parquet"
    result.to_parquet(output_path)
    
    logger.info(f"\nON {asset}: 总计 {len(result):,} 条")
    logger.info(f"   范围: {result['timestamp'].min()} ~ {result['timestamp'].max()}")
    logger.info(f"   保存: {output_path}")
    
    return len(result)


def main():
    parser = argparse.ArgumentParser(description='批量下载 Binance 历史数据')
    
    parser.add_argument('--years', type=int, default=2, help='下载最近 N 年数据')
    parser.add_argument('--start', type=str, help='开始月份 (格式: 2022-01)')
    parser.add_argument('--end', type=str, help='结束月份 (格式: 2025-01)')
    parser.add_argument('--assets', type=str, default='BTC,ETH,SOL', help='资产列表')
    parser.add_argument('--output', type=str, default='./training_data/raw', help='输出目录')
    
    args = parser.parse_args()
    
    # 计算日期范围
    now = datetime.now()
    
    if args.start and args.end:
        start_year, start_month = map(int, args.start.split('-'))
        end_year, end_month = map(int, args.end.split('-'))
    else:
        end_year = now.year
        end_month = now.month - 1  # 上个月（当月可能不完整）
        if end_month == 0:
            end_month = 12
            end_year -= 1
        
        start_year = end_year - args.years
        start_month = end_month
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载资产
    assets = [a.strip() for a in args.assets.split(',')]
    
    total = 0
    for asset in assets:
        count = download_asset(asset, start_year, start_month, end_year, end_month, output_dir)
        total += count
    
    # 汇总
    print("\n" + "="*60)
    print("下载完成!")
    print("="*60)
    
    for asset in assets:
        file_path = output_dir / f"{asset}_60m.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            days = len(df) / 24
            print(f"  {asset}: {len(df):,} 条 ({days:.0f} 天)")
    
    print(f"\n总计: {total:,} 条数据")
    print(f"保存目录: {output_dir}")
    
    # 检查是否满足需求
    print("\n--- 训练需求检查 ---")
    for asset in assets:
        file_path = output_dir / f"{asset}_60m.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            dt_ok = "ON" if len(df) >= 5000 else "OFF"
            drl_ok = "ON" if len(df) >= 3000 else "OFF"
            print(f"  {asset}: DT v3.2 {dt_ok} | DRL v5.5 {drl_ok}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
================================================================================
HMATS 增强版特征工程 v2
================================================================================

整合所有数据源（处理日线->小时线合并）:
- OHLCV 基础特征 (1H)
- Funding Rate 特征 (1D -> forward fill 到 1H)
- Open Interest 特征 (1D -> forward fill 到 1H)
- 期货溢价特征 (1D -> forward fill 到 1H)

特征总维度: 48
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger("Features-v2")


class DailyToHourlyMerger:
    """
    日线数据 -> 小时线合并器
    
    策略: 将日线数据 forward fill 到每个小时
    例: 1月1日的日线数据会填充到 1月1日 00:00 ~ 23:00 的每个小时
    """
    
    @staticmethod
    def find_time_col(df: pd.DataFrame) -> Optional[str]:
        """找到时间列"""
        for col in ['timestamp', 'time', 'date', 'datetime', 'fundingtime', 't']:
            if col in df.columns:
                return col
        return None
    
    @staticmethod
    def merge_daily_to_hourly(hourly_df: pd.DataFrame, 
                               daily_df: pd.DataFrame,
                               value_col: str,
                               output_col: str = None) -> pd.DataFrame:
        """
        将日线数据合并到小时线
        
        Args:
            hourly_df: 小时线 DataFrame (必须有 'timestamp' 列)
            daily_df: 日线 DataFrame
            value_col: 日线中要合并的值列名
            output_col: 输出列名 (默认与 value_col 相同)
        
        Returns:
            合并后的 hourly_df
        """
        if output_col is None:
            output_col = value_col
        
        if daily_df is None or len(daily_df) == 0:
            hourly_df[output_col] = 0.0
            return hourly_df
        
        # 找到日线的时间列
        time_col = DailyToHourlyMerger.find_time_col(daily_df)
        if time_col is None:
            logger.warning(f"日线数据无时间列，跳过 {value_col}")
            hourly_df[output_col] = 0.0
            return hourly_df
        
        # 找到值列
        value_col_actual = None
        for col in [value_col, value_col.lower(), value_col.upper()]:
            if col in daily_df.columns:
                value_col_actual = col
                break
        
        if value_col_actual is None:
            logger.warning(f"日线数据无 {value_col} 列，跳过")
            hourly_df[output_col] = 0.0
            return hourly_df
        
        # 准备日线数据
        daily_df = daily_df.copy()
        daily_df['_date'] = pd.to_datetime(daily_df[time_col]).dt.date
        daily_df['_value'] = pd.to_numeric(daily_df[value_col_actual], errors='coerce').fillna(0)
        
        # 创建日期到值的映射
        date_to_value = daily_df.groupby('_date')['_value'].last().to_dict()
        
        # 在小时线上应用
        if 'timestamp' in hourly_df.columns:
            hourly_dates = pd.to_datetime(hourly_df['timestamp']).dt.date
            hourly_df[output_col] = hourly_dates.map(date_to_value).fillna(method='ffill').fillna(0)
        else:
            hourly_df[output_col] = 0.0
        
        return hourly_df


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str = 'training_data'):
        self.data_dir = Path(data_dir)
    
    def load_spot_1h(self, asset: str) -> Optional[pd.DataFrame]:
        """加载现货 OHLCV (1H)"""
        path = self.data_dir / 'raw' / f'{asset}_60m.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        return None
    
    def load_futures_1d(self, asset: str) -> Optional[pd.DataFrame]:
        """加载期货 OHLCV (1D 日线)"""
        path = self.data_dir / 'futures_daily' / f'{asset}_futures_1d.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()
            return df
        return None
    
    def load_funding_1d(self, asset: str) -> Optional[pd.DataFrame]:
        """加载 Funding Rate (1D 日线)"""
        path = self.data_dir / 'funding_daily' / f'{asset}_funding_1d.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()
            return df
        return None
    
    def load_oi_1d(self, asset: str) -> Optional[pd.DataFrame]:
        """加载 Open Interest (1D 日线)"""
        path = self.data_dir / 'oi_daily' / f'{asset}_oi_1d.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df.columns = df.columns.str.lower()
            return df
        return None


class EnhancedFeatureEngineer:
    """
    增强版特征工程
    
    特征维度: 48
    - OHLCV 基础: 40
    - Funding Rate: 3
    - Open Interest: 3
    - 期货溢价: 2
    """
    
    FEATURE_DIM = 48
    
    def __init__(self, data_dir: str = 'training_data'):
        self.loader = DataLoader(data_dir)
        self.merger = DailyToHourlyMerger()
    
    def load_and_merge(self, asset: str) -> pd.DataFrame:
        """加载并合并所有数据源"""
        
        logger.info(f"加载 {asset} 数据...")
        
        # 1. 加载现货 1H (主数据)
        df = self.loader.load_spot_1h(asset)
        if df is None:
            raise ValueError(f"无法加载 {asset} 现货数据")
        logger.info(f"  现货 1H: {len(df)} 行")
        
        # 2. 合并 Funding Rate (日线 -> 小时)
        funding_df = self.loader.load_funding_1d(asset)
        if funding_df is not None:
            logger.info(f"  Funding 1D: {len(funding_df)} 行 -> 合并到小时")
            
            # 尝试多种可能的列名
            for rate_col in ['fundingrate', 'funding_rate', 'rate', 'funding']:
                if rate_col in funding_df.columns:
                    df = self.merger.merge_daily_to_hourly(df, funding_df, rate_col, 'funding_rate')
                    break
            else:
                df['funding_rate'] = 0.0
        else:
            logger.warning(f"  Funding: 无数据")
            df['funding_rate'] = 0.0
        
        # 3. 合并 Open Interest (日线 -> 小时)
        oi_df = self.loader.load_oi_1d(asset)
        if oi_df is not None:
            logger.info(f"  OI 1D: {len(oi_df)} 行 -> 合并到小时")
            
            for oi_col in ['openinterest', 'open_interest', 'oi', 'sumopeninterest', 'sumOpenInterest']:
                if oi_col in oi_df.columns:
                    df = self.merger.merge_daily_to_hourly(df, oi_df, oi_col, 'open_interest')
                    break
            else:
                df['open_interest'] = 0.0
        else:
            logger.warning(f"  OI: 无数据")
            df['open_interest'] = 0.0
        
        # 4. 合并期货溢价 (日线 -> 小时)
        futures_df = self.loader.load_futures_1d(asset)
        if futures_df is not None and 'close' in futures_df.columns:
            logger.info(f"  期货 1D: {len(futures_df)} 行 -> 计算溢价")
            
            # 找时间列
            time_col = self.merger.find_time_col(futures_df)
            if time_col:
                futures_df = futures_df.copy()
                futures_df['_date'] = pd.to_datetime(futures_df[time_col]).dt.date
                futures_df['futures_close'] = pd.to_numeric(futures_df['close'], errors='coerce')
                date_to_futures = futures_df.groupby('_date')['futures_close'].last().to_dict()
                
                if 'timestamp' in df.columns:
                    hourly_dates = pd.to_datetime(df['timestamp']).dt.date
                    df['futures_close_daily'] = hourly_dates.map(date_to_futures).fillna(method='ffill')
                    
                    # 计算溢价 = (期货 - 现货) / 现货
                    df['futures_premium'] = ((df['futures_close_daily'] - df['close']) / df['close']).fillna(0).clip(-0.1, 0.1)
                    df = df.drop(columns=['futures_close_daily'])
                else:
                    df['futures_premium'] = 0.0
            else:
                df['futures_premium'] = 0.0
        else:
            logger.warning(f"  期货: 无数据")
            df['futures_premium'] = 0.0
        
        return df
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        
        df = df.copy()
        
        # ========== OHLCV 基础特征 (40) ==========
        
        # 收益率 (10)
        for p in [1, 4, 12, 24, 48, 72, 96, 168, 336, 720]:
            df[f'ret_{p}h'] = df['close'].pct_change(p).fillna(0).clip(-0.5, 0.5)
        
        # 波动率 (5)
        hourly_ret = df['close'].pct_change(1).fillna(0)
        for w in [12, 24, 48, 96, 168]:
            df[f'vol_{w}h'] = hourly_ret.rolling(w, min_periods=1).std().fillna(0) * np.sqrt(24*365)
        
        # 动量 (5)
        for p in [12, 24, 48, 96, 168]:
            df[f'mom_{p}h'] = df['close'].pct_change(p).fillna(0).clip(-1, 1)
        
        # Z-score (4)
        for w in [24, 48, 168, 336]:
            ma = df['close'].rolling(w, min_periods=1).mean()
            std = df['close'].rolling(w, min_periods=1).std()
            df[f'zscore_{w}h'] = ((df['close'] - ma) / (std + 1e-10)).fillna(0).clip(-5, 5)
        
        # 技术指标 (6)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100 - 0.5
        
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ((ema12 - ema26) / df['close']).fillna(0).clip(-0.1, 0.1)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        bb_ma = df['close'].rolling(20, min_periods=1).mean()
        bb_std = df['close'].rolling(20, min_periods=1).std()
        df['bb_position'] = ((df['close'] - bb_ma) / (2 * bb_std + 1e-10)).fillna(0).clip(-1, 1)
        df['trend_strength'] = (df['close'].rolling(24, min_periods=1).mean() / 
                                df['close'].rolling(168, min_periods=1).mean() - 1).fillna(0).clip(-0.3, 0.3)
        
        # 成交量 (4)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vol_ma = df['volume'].rolling(24, min_periods=1).mean()
            df['vol_zscore_v'] = ((df['volume'] - vol_ma) / (vol_ma.std() + 1e-10)).fillna(0).clip(-5, 5)
            df['vol_trend'] = (df['volume'].rolling(24, min_periods=1).mean() / 
                              df['volume'].rolling(168, min_periods=1).mean() - 1).fillna(0).clip(-2, 2)
            df['vol_spike'] = (df['volume'] / (vol_ma + 1e-10)).fillna(1).clip(0, 10) - 1
            df['vol_momentum'] = df['volume'].pct_change(24).fillna(0).clip(-2, 2)
        else:
            df['vol_zscore_v'] = df['vol_trend'] = df['vol_spike'] = df['vol_momentum'] = 0
        
        # 方向性 (6)
        df['bullish'] = (df['ret_24h'] > 0.02).astype(float)
        df['bearish'] = (df['ret_24h'] < -0.02).astype(float)
        df['high_vol'] = (df['vol_24h'] > df['vol_24h'].rolling(168, min_periods=1).mean()).astype(float)
        df['trending_up'] = (df['mom_168h'] > 0.05).astype(float)
        df['trending_down'] = (df['mom_168h'] < -0.05).astype(float)
        df['range_bound'] = ((df['mom_168h'].abs() < 0.05) & (df['vol_24h'] < 0.5)).astype(float)
        
        # ========== Funding Rate 特征 (3) ==========
        # funding_rate 已在 merge 时创建
        df['funding_rate'] = df['funding_rate'].fillna(0)
        df['funding_rate_ma'] = df['funding_rate'].rolling(7, min_periods=1).mean()  # 7日MA (因为是日线)
        df['funding_rate_cumsum'] = df['funding_rate'].rolling(7*24, min_periods=1).sum()  # 7天累计
        
        # ========== Open Interest 特征 (3) ==========
        df['open_interest'] = df['open_interest'].fillna(0)
        oi_max = df['open_interest'].rolling(720, min_periods=1).max()
        df['oi_normalized'] = (df['open_interest'] / (oi_max + 1e-10)).fillna(0).clip(0, 2)
        df['oi_change'] = df['open_interest'].pct_change(24).fillna(0).clip(-1, 1)
        oi_ma = df['open_interest'].rolling(168, min_periods=1).mean()
        oi_std = df['open_interest'].rolling(168, min_periods=1).std()
        df['oi_zscore'] = ((df['open_interest'] - oi_ma) / (oi_std + 1e-10)).fillna(0).clip(-5, 5)
        
        # ========== 期货溢价特征 (2) ==========
        df['futures_premium'] = df['futures_premium'].fillna(0)
        df['basis'] = df['futures_premium'].rolling(24, min_periods=1).mean()
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名 (48维)"""
        cols = []
        
        # OHLCV 基础 (40)
        for p in [1, 4, 12, 24, 48, 72, 96, 168, 336, 720]:
            cols.append(f'ret_{p}h')
        for w in [12, 24, 48, 96, 168]:
            cols.append(f'vol_{w}h')
        for p in [12, 24, 48, 96, 168]:
            cols.append(f'mom_{p}h')
        for w in [24, 48, 168, 336]:
            cols.append(f'zscore_{w}h')
        cols.extend(['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'trend_strength'])
        cols.extend(['vol_zscore_v', 'vol_trend', 'vol_spike', 'vol_momentum'])
        cols.extend(['bullish', 'bearish', 'high_vol', 'trending_up', 'trending_down', 'range_bound'])
        
        # Funding Rate (3)
        cols.extend(['funding_rate', 'funding_rate_ma', 'funding_rate_cumsum'])
        
        # Open Interest (3)
        cols.extend(['oi_normalized', 'oi_change', 'oi_zscore'])
        
        # 期货溢价 (2)
        cols.extend(['futures_premium', 'basis'])
        
        return cols  # 40 + 3 + 3 + 2 = 48
    
    def prepare_training_data(self, asset: str) -> pd.DataFrame:
        """准备训练数据"""
        
        # 加载合并
        df = self.load_and_merge(asset)
        
        # 计算特征
        df = self.compute_features(df)
        
        # 删除前 720 行 (特征计算需要历史)
        df = df.iloc[720:].reset_index(drop=True)
        
        logger.info(f"ON {asset} 特征准备完成: {len(df)} 行, {len(self.get_feature_columns())} 维特征")
        
        return df


def prepare_all_assets(data_dir: str = 'training_data', 
                       output_dir: str = 'training_data/processed') -> Dict[str, pd.DataFrame]:
    """准备所有资产的训练数据"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    engineer = EnhancedFeatureEngineer(data_dir)
    results = {}
    
    for asset in ['SOL', 'BTC', 'ETH']:
        try:
            df = engineer.prepare_training_data(asset)
            
            path = output_path / f'{asset}_features.parquet'
            df.to_parquet(path)
            logger.info(f"💾 保存: {path}")
            
            results[asset] = df
            
        except Exception as e:
            logger.error(f"OFF {asset} 处理失败: {e}")
            continue
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_all_assets()

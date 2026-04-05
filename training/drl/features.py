"""
特征工程模块 - 128维主特征 + Cross-Asset特征
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class FeatureEngineer:
    """v5.5 特征工程"""
    
    def __init__(self, state_dim: int = 128, cross_asset_dim: int = 16):
        self.state_dim = state_dim
        self.cross_asset_dim = cross_asset_dim
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 128 维主特征"""
        df = df.copy()
        
        # 收益率 (16)
        for p in [1, 2, 4, 6, 8, 12, 24, 48, 72, 96, 120, 168, 336, 504, 720, 1000]:
            df[f'ret_{p}h'] = df['close'].pct_change(p).fillna(0).clip(-0.5, 0.5)
        
        # 波动率 (10)
        hourly_ret = df['close'].pct_change(1).fillna(0)
        for w in [6, 12, 24, 48, 72, 96, 120, 168, 336, 720]:
            df[f'vol_{w}h'] = hourly_ret.rolling(w).std().fillna(0) * np.sqrt(24*365)
        
        # 波动率指标 (6)
        df['vol_ratio_s'] = (df['vol_24h'] / (df['vol_168h'] + 1e-10)).clip(0, 5)
        df['vol_ratio_l'] = (df['vol_72h'] / (df['vol_336h'] + 1e-10)).clip(0, 5)
        df['vol_regime'] = (df['vol_24h'] > df['vol_24h'].rolling(168).mean()).astype(float)
        df['vol_trend'] = df['vol_24h'].diff(24).fillna(0).clip(-1, 1)
        df['vol_accel'] = df['vol_trend'].diff(12).fillna(0).clip(-0.5, 0.5)
        df['vol_skew'] = hourly_ret.rolling(168).skew().fillna(0).clip(-3, 3)
        
        # 动量 (4) - only long-horizon (short-horizon mom_Xh ≡ ret_Xh, r=1.000)
        for p in [120, 168, 336, 720]:
            df[f'mom_{p}h'] = df['close'].pct_change(p).fillna(0).clip(-1, 1)
        
        # Z-score (6)
        for w in [24, 48, 96, 168, 336, 720]:
            ma = df['close'].rolling(w).mean()
            std = df['close'].rolling(w).std()
            df[f'zscore_{w}h'] = ((df['close'] - ma) / (std + 1e-10)).fillna(0).clip(-5, 5)
        
        # RSI (4)
        for period in [7, 14, 21, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = (100 - (100 / (1 + rs))).fillna(50) / 100 - 0.5
        
        # MACD (4)
        for fast, slow in [(6, 13), (12, 26)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = ((ema_fast - ema_slow) / df['close']).fillna(0).clip(-0.1, 0.1)
            df[f'macd_{fast}_{slow}_sig'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        
        # Bollinger (4)
        for w in [20, 50]:
            bb_ma = df['close'].rolling(w).mean()
            bb_std = df['close'].rolling(w).std()
            df[f'bb_pos_{w}'] = ((df['close'] - bb_ma) / (2 * bb_std + 1e-10)).fillna(0).clip(-1, 1)
            df[f'bb_width_{w}'] = (4 * bb_std / bb_ma).fillna(0).clip(0, 1)
        
        # ATR (3)
        for period in [7, 14, 28]:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = (tr.rolling(period).mean() / df['close']).fillna(0).clip(0, 0.2)
        
        # 成交量 (6)
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(24).mean()
            df['vol_zscore'] = ((df['volume'] - vol_ma) / (vol_ma.std() + 1e-10)).fillna(0).clip(-5, 5)
            df['vol_spike'] = (df['volume'] / (vol_ma + 1e-10)).fillna(1).clip(0, 10) - 1
            df['vol_trend_v'] = (df['volume'].rolling(24).mean() / df['volume'].rolling(168).mean() - 1).fillna(0).clip(-1, 1)
            df['vol_price_corr'] = df['close'].pct_change().rolling(24).corr(df['volume'].pct_change()).fillna(0)
            df['vol_ret_int'] = df['vol_spike'] * df['ret_4h']
            df['vol_mom'] = df['volume'].pct_change(24).fillna(0).clip(-2, 2)
        else:
            for col in ['vol_zscore', 'vol_spike', 'vol_trend_v', 'vol_price_corr', 'vol_ret_int', 'vol_mom']:
                df[col] = 0
        
        # 做空特征 (18)
        df['fear_1h'] = (df['ret_1h'] < -0.015).astype(float)
        df['fear_4h'] = (df['ret_4h'] < -0.025).astype(float)
        df['fear_extreme'] = (df['ret_4h'] < -0.05).astype(float)
        df['panic'] = (df['ret_4h'] < -0.07).astype(float)
        df['capitulation'] = ((df['ret_4h'] < -0.08) & (df['vol_spike'] > 2)).astype(float)
        df['breakdown_20'] = (df['close'] < df['close'].rolling(20).min().shift(1)).astype(float)
        df['breakdown_50'] = (df['close'] < df['close'].rolling(50).min().shift(1)).astype(float)
        df['breakdown_100'] = (df['close'] < df['close'].rolling(100).min().shift(1)).astype(float)
        df['vol_expansion'] = ((df['vol_24h'] > df['vol_168h'] * 1.5) & (df['ret_4h'] < 0)).astype(float)
        df['consecutive_red'] = sum((df['close'].shift(i-1) < df['open'].shift(i-1)).astype(float) * 0.125 for i in range(1, 8))
        df['consecutive_red'] = df['consecutive_red'].clip(0, 1)
        ma7 = df['close'].rolling(7).mean()
        ma25 = df['close'].rolling(25).mean()
        df['death_cross'] = ((ma7 < ma25) & (ma7.shift(1) > ma25.shift(1))).astype(float)
        df['ma_div'] = ((ma7 - ma25) / ma25).fillna(0).clip(-0.2, 0.2)
        df['below_ma50'] = (df['close'] < df['close'].rolling(50).mean()).astype(float)
        df['below_ma200'] = (df['close'] < df['close'].rolling(200).mean()).astype(float)
        df['dist_high'] = (df['close'] / df['high'].rolling(20).max() - 1).fillna(0).clip(-0.3, 0)
        df['dist_low'] = (df['close'] / df['low'].rolling(20).min() - 1).fillna(0).clip(0, 0.3)
        df['short_score'] = (df['fear_4h'] * 0.12 + df['breakdown_20'] * 0.15 + df['breakdown_50'] * 0.1 +
                            df['death_cross'] * 0.12 + df['vol_expansion'] * 0.12 + df['below_ma50'] * 0.08 +
                            df['consecutive_red'] * 0.1 + df['panic'] * 0.13).clip(0, 1)
        
        # 做多特征 (6)
        df['greed_1h'] = (df['ret_1h'] > 0.015).astype(float)
        df['greed_4h'] = (df['ret_4h'] > 0.025).astype(float)
        df['breakout_20'] = (df['close'] > df['close'].rolling(20).max().shift(1)).astype(float)
        df['golden_cross'] = ((ma7 > ma25) & (ma7.shift(1) < ma25.shift(1))).astype(float)
        df['above_ma50'] = (df['close'] > df['close'].rolling(50).mean()).astype(float)
        df['long_score'] = (df['greed_4h'] * 0.15 + df['breakout_20'] * 0.2 + df['golden_cross'] * 0.15 +
                           df['above_ma50'] * 0.2 + (df['rsi_14'] < -0.2).astype(float) * 0.15 + df['greed_1h'] * 0.15).clip(0, 1)
        
        # 风险 (10)
        df['dd_local'] = (df['close'] / df['close'].rolling(24).max() - 1).fillna(0).clip(-0.3, 0)
        df['dd_medium'] = (df['close'] / df['close'].rolling(168).max() - 1).fillna(0).clip(-0.5, 0)
        df['dd_long'] = (df['close'] / df['close'].rolling(720).max() - 1).fillna(0).clip(-0.7, 0)
        df['recovery'] = ((df['close'] - df['close'].rolling(24).min()) / 
                         (df['close'].rolling(24).max() - df['close'].rolling(24).min() + 1e-10)).fillna(0.5).clip(0, 1)
        df['vol_regime_f'] = (df['vol_24h'] / df['vol_24h'].rolling(720).mean()).fillna(1).clip(0.5, 3) - 1
        df['extreme_move'] = ((abs(df['ret_4h']) > 0.05) | (abs(df['ret_24h']) > 0.12)).astype(float)
        df['gap_risk'] = abs(df['open'] / df['close'].shift(1) - 1).fillna(0).clip(0, 0.1)
        rolling_ret = df['close'].pct_change(1)
        df['roll_sharpe'] = (rolling_ret.rolling(168).mean() / (rolling_ret.rolling(168).std() + 1e-10) * np.sqrt(168)).fillna(0).clip(-3, 3)
        downside = rolling_ret.where(rolling_ret < 0, 0)
        df['roll_sortino'] = (rolling_ret.rolling(168).mean() / (downside.rolling(168).std() + 1e-10) * np.sqrt(168)).fillna(0).clip(-3, 3)
        df['roll_cvar'] = rolling_ret.rolling(168).apply(lambda x: x[x <= x.quantile(0.05)].mean() if len(x) > 10 and len(x[x <= x.quantile(0.05)]) > 0 else x.quantile(0.05), raw=False).fillna(0).clip(-0.2, 0)
        
        # 趋势 (6)
        df['trend_str'] = (df['close'].rolling(24).mean() / df['close'].rolling(168).mean() - 1).fillna(0).clip(-0.3, 0.3)
        df['trend_accel'] = df['trend_str'].diff(24).fillna(0).clip(-0.1, 0.1)
        df['trend_down'] = (df['close'] < df['close'].rolling(72).mean()).astype(float)
        df['trend_up'] = (df['close'] > df['close'].rolling(72).mean()).astype(float)
        df['higher_high'] = (df['high'] > df['high'].rolling(24).max().shift(1)).astype(float)
        df['lower_low'] = (df['low'] < df['low'].rolling(24).min().shift(1)).astype(float)
        
        return df.fillna(0)
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列"""
        return [
            # 收益率 (16)
            *[f'ret_{p}h' for p in [1, 2, 4, 6, 8, 12, 24, 48, 72, 96, 120, 168, 336, 504, 720, 1000]],
            # 波动率 (10)
            *[f'vol_{w}h' for w in [6, 12, 24, 48, 72, 96, 120, 168, 336, 720]],
            # 波动率指标 (6)
            'vol_ratio_s', 'vol_ratio_l', 'vol_regime', 'vol_trend', 'vol_accel', 'vol_skew',
            # 动量 (4 - long-horizon only, short ≡ ret_Xh)
            *[f'mom_{p}h' for p in [120, 168, 336, 720]],
            # Z-score (6)
            *[f'zscore_{w}h' for w in [24, 48, 96, 168, 336, 720]],
            # RSI (4)
            *[f'rsi_{p}' for p in [7, 14, 21, 28]],
            # MACD (4)
            'macd_6_13', 'macd_6_13_sig', 'macd_12_26', 'macd_12_26_sig',
            # Bollinger (4)
            'bb_pos_20', 'bb_width_20', 'bb_pos_50', 'bb_width_50',
            # ATR (3)
            'atr_7', 'atr_14', 'atr_28',
            # 成交量 (6)
            'vol_zscore', 'vol_spike', 'vol_trend_v', 'vol_price_corr', 'vol_ret_int', 'vol_mom',
            # 做空 (18)
            'fear_1h', 'fear_4h', 'fear_extreme', 'panic', 'capitulation', 'breakdown_20', 'breakdown_50',
            'breakdown_100', 'vol_expansion', 'consecutive_red', 'death_cross', 'ma_div', 'below_ma50',
            'below_ma200', 'dist_high', 'dist_low', 'short_score',
            # 做多 (6)
            'greed_1h', 'greed_4h', 'breakout_20', 'golden_cross', 'above_ma50', 'long_score',
            # 风险 (10)
            'dd_local', 'dd_medium', 'dd_long', 'recovery', 'vol_regime_f', 'extreme_move',
            'gap_risk', 'roll_sharpe', 'roll_sortino', 'roll_cvar',
            # 趋势 (6)
            'trend_str', 'trend_accel', 'trend_down', 'trend_up', 'higher_high', 'lower_low',
        ][:self.state_dim - self.cross_asset_dim]
    
    def compute_cross_asset_features(self, btc_df: pd.DataFrame = None, 
                                      eth_df: pd.DataFrame = None,
                                      sol_df: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """计算 Cross-Asset 特征"""
        result = {'btc_features': np.zeros(8), 'eth_features': np.zeros(8), 'lead_lag': np.zeros(4)}
        
        if btc_df is not None and len(btc_df) >= 24:
            result['btc_features'] = self._compute_asset_stats(btc_df)
        
        if eth_df is not None and len(eth_df) >= 24:
            result['eth_features'] = self._compute_asset_stats(eth_df)
        
        if btc_df is not None and sol_df is not None:
            result['lead_lag'] = self._compute_lead_lag(btc_df, sol_df)
        
        return result
    
    def _compute_asset_stats(self, df: pd.DataFrame) -> np.ndarray:
        """计算资产统计"""
        if len(df) < 24:
            return np.zeros(8)
        
        ret = df['close'].pct_change()
        return np.array([
            ret.iloc[-4:].sum(),
            ret.iloc[-24:].sum(),
            ret.iloc[-24:].std() * np.sqrt(24),
            ret.iloc[-168:].std() * np.sqrt(168) if len(ret) >= 168 else 0,
            (df['close'].iloc[-1] / df['close'].rolling(50).mean().iloc[-1] - 1) if len(df) >= 50 else 0,
            1 if ret.iloc[-4:].sum() < -0.02 else 0,
            1 if ret.iloc[-4:].sum() > 0.02 else 0,
            (df['volume'].iloc[-24:].mean() / df['volume'].iloc[-168:].mean() - 1) if 'volume' in df.columns and len(df) >= 168 else 0,
        ], dtype=np.float32)
    
    def _compute_lead_lag(self, btc_df: pd.DataFrame, sol_df: pd.DataFrame) -> np.ndarray:
        """计算领先滞后"""
        if len(btc_df) < 24 or len(sol_df) < 24:
            return np.zeros(4)
        
        btc_ret = btc_df['close'].pct_change().iloc[-24:]
        sol_ret = sol_df['close'].pct_change().iloc[-24:]
        min_len = min(len(btc_ret), len(sol_ret))
        btc_ret, sol_ret = btc_ret.iloc[-min_len:], sol_ret.iloc[-min_len:]
        
        return np.array([
            btc_ret.corr(sol_ret) if len(btc_ret) > 5 else 0,
            btc_ret.iloc[:-1].corr(sol_ret.iloc[1:]) if len(btc_ret) > 6 else 0,
            btc_ret.iloc[:-4].corr(sol_ret.iloc[4:]) if len(btc_ret) > 8 else 0,
            btc_ret.sum() - sol_ret.sum(),
        ], dtype=np.float32)


def get_regime_probs(df: pd.DataFrame, idx: int) -> np.ndarray:
    """计算软 6-Regime 概率"""
    if idx < 168:
        return np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1], dtype=np.float32)
    
    try:
        ret_24h = df['close'].iloc[idx] / df['close'].iloc[idx-24] - 1
        ret_168h = df['close'].iloc[idx] / df['close'].iloc[idx-168] - 1
        vol_24h = df.iloc[idx].get('vol_24h', 0.5)
        vol_168h = df.iloc[idx].get('vol_168h', 0.5)
        vol_ratio = vol_24h / (vol_168h + 1e-10)
        
        # 6-Regime 软概率计算
        strong_bear = max(0, (-ret_168h - 0.05) * 10) * max(0, vol_ratio - 1)
        moderate_bear = max(0, -ret_168h * 5) * (1 - strong_bear * 0.5)
        strong_bull = max(0, (ret_168h - 0.05) * 10) * max(0, 1.5 - vol_ratio)
        moderate_bull = max(0, ret_168h * 5) * (1 - strong_bull * 0.5)
        volatile = max(0, vol_ratio - 1.3) * 3
        neutral = max(0, 1 - abs(ret_168h) * 8 - volatile * 0.3)
        
        scores = np.array([strong_bear, moderate_bear, neutral, volatile, moderate_bull, strong_bull])
        return (np.exp(scores) / np.exp(scores).sum()).astype(np.float32)
    except:
        return np.array([0.1, 0.1, 0.3, 0.2, 0.2, 0.1], dtype=np.float32)

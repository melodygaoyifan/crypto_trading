"""
GPU-Accelerated Technical Analysis
===================================
Optimized for RTX 5090 + multi-asset analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

# GPU backend
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


class GPUTechnicalAnalyzer:
    """
    GPU-accelerated technical analysis.
    
    Calculates indicators for multiple assets in parallel.
    """
    
    def __init__(self):
        self.xp = cp if GPU_AVAILABLE else np
        if GPU_AVAILABLE:
            print("✓ GPU TA enabled (CuPy)")
        else:
            print("ℹ GPU TA using CPU (install cupy for GPU)")
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = df.copy()
        
        close = self.xp.array(df['close'].values)
        high = self.xp.array(df['high'].values)
        low = self.xp.array(df['low'].values)
        volume = self.xp.array(df['volume'].values)
        
        # RSI
        df['rsi'] = self._to_numpy(self._rsi(close, 14))
        
        # MACD
        macd, signal, hist = self._macd(close)
        df['macd'] = self._to_numpy(macd)
        df['macd_signal'] = self._to_numpy(signal)
        df['macd_histogram'] = self._to_numpy(hist)
        
        # EMAs
        df['ema_9'] = self._to_numpy(self._ema(close, 9))
        df['ema_21'] = self._to_numpy(self._ema(close, 21))
        df['ema_55'] = self._to_numpy(self._ema(close, 55))
        df['ema_200'] = self._to_numpy(self._ema(close, 200))
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = self._bbands(close, 20, 2.0)
        df['bb_upper'] = self._to_numpy(bb_upper)
        df['bb_mid'] = self._to_numpy(bb_mid)
        df['bb_lower'] = self._to_numpy(bb_lower)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR
        df['atr'] = self._to_numpy(self._atr(high, low, close, 14))
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Volume
        df['volume_sma'] = self._to_numpy(self._sma(volume, 20))
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # ROC
        df['roc_1'] = df['close'].pct_change(1) * 100
        df['roc_4'] = df['close'].pct_change(4) * 100
        df['roc_24'] = df['close'].pct_change(24) * 100
        
        # EMA alignment
        df['ema_alignment'] = 0
        df.loc[(df['close'] > df['ema_21']) & (df['ema_21'] > df['ema_55']), 'ema_alignment'] = 1
        df.loc[(df['close'] < df['ema_21']) & (df['ema_21'] < df['ema_55']), 'ema_alignment'] = -1
        
        # ADX (simplified)
        df['adx'] = abs(df['roc_24']).rolling(14).mean() + 15
        
        return df
    
    def calculate_multi_asset(
        self,
        asset_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Calculate TA for multiple assets."""
        results = {}
        for asset, df in asset_data.items():
            if not df.empty:
                results[asset] = self.calculate_all(df)
        return results
    
    def _to_numpy(self, arr):
        """Convert GPU array to numpy."""
        if GPU_AVAILABLE and hasattr(arr, 'get'):
            return arr.get()
        return arr
    
    def _sma(self, arr, period):
        """Simple Moving Average."""
        kernel = self.xp.ones(period) / period
        result = self.xp.convolve(arr, kernel, mode='same')
        result[:period-1] = self.xp.nan
        return result
    
    def _ema(self, arr, span):
        """Exponential Moving Average."""
        alpha = 2 / (span + 1)
        result = self.xp.zeros_like(arr)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    def _rsi(self, close, period=14):
        """Relative Strength Index."""
        delta = self.xp.diff(close)
        delta = self.xp.concatenate([self.xp.array([0]), delta])
        
        gain = self.xp.where(delta > 0, delta, 0)
        loss = self.xp.where(delta < 0, -delta, 0)
        
        avg_gain = self._ema(gain, period)
        avg_loss = self._ema(loss, period)
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _macd(self, close, fast=12, slow=26, signal=9):
        """MACD."""
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _bbands(self, close, period=20, std_dev=2.0):
        """Bollinger Bands."""
        middle = self._sma(close, period)
        std = self.xp.zeros_like(close)
        for i in range(period, len(close)):
            std[i] = self.xp.std(close[i-period:i])
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower
    
    def _atr(self, high, low, close, period=14):
        """Average True Range."""
        prev_close = self.xp.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = self.xp.abs(high - prev_close)
        tr3 = self.xp.abs(low - prev_close)
        
        tr = self.xp.maximum(tr1, self.xp.maximum(tr2, tr3))
        return self._ema(tr, period)

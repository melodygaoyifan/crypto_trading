#!/usr/bin/env python3
"""
================================================================================
HMATS 6-Regime GMM v2 - 整合日线数据
================================================================================

特征 (25维):
- 动量/收益 (6)
- 波动率 (4)
- Funding Rate (3) - 日线合并
- Open Interest (3) - 日线合并
- 期货溢价 (2) - 日线合并
- 技术指标 (4)
- 成交量 (3)
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, NamedTuple
from pathlib import Path
import logging
import pickle

logger = logging.getLogger("GMM-v2")

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class Regime:
    STRONG_BEAR = 0
    MODERATE_BEAR = 1
    NEUTRAL = 2
    VOLATILE_CHOP = 3
    MODERATE_BULL = 4
    STRONG_BULL = 5
    
    NAMES = ['STRONG_BEAR', 'MODERATE_BEAR', 'NEUTRAL', 
             'VOLATILE_CHOP', 'MODERATE_BULL', 'STRONG_BULL']
    
    SHORT_BIAS = {0: -0.25, 1: -0.10, 2: 0.00, 3: 0.00, 4: +0.10, 5: +0.15}
    
    @classmethod
    def to_name(cls, regime_id: int) -> str:
        return cls.NAMES[regime_id] if 0 <= regime_id < 6 else 'UNKNOWN'


class RegimeState(NamedTuple):
    regime_id: int
    regime_name: str
    probabilities: np.ndarray
    confidence: float
    features: np.ndarray


class GMMFeatureExtractor:
    """GMM 特征提取器 (25维)"""
    
    FEATURE_DIM = 25
    
    def extract(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """提取单点特征"""
        
        if idx < 720:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)
        
        try:
            # 动量/收益 (6)
            ret_24h = df['close'].iloc[idx] / df['close'].iloc[idx-24] - 1
            ret_7d = df['close'].iloc[idx] / df['close'].iloc[idx-168] - 1
            ret_30d = df['close'].iloc[idx] / df['close'].iloc[idx-720] - 1 if idx >= 720 else 0
            mom_24h = ret_24h
            mom_7d = ret_7d
            mom_consistency = float(np.sign(ret_24h) == np.sign(ret_7d))
            
            # 波动率 (4)
            returns = df['close'].pct_change().iloc[idx-168:idx]
            vol_7d = returns.std() * np.sqrt(24 * 365) if len(returns) > 10 else 0
            returns_30d = df['close'].pct_change().iloc[idx-720:idx] if idx >= 720 else returns
            vol_30d = returns_30d.std() * np.sqrt(24 * 365) if len(returns_30d) > 10 else 0
            vol_ratio = vol_7d / (vol_30d + 1e-10)
            vol_of_vol = returns.rolling(24).std().std() if len(returns) > 24 else 0
            
            # Funding Rate (3) - 从合并后的列读取
            funding_rate = df['funding_rate'].iloc[idx] if 'funding_rate' in df.columns else 0
            funding_ma = df['funding_rate_ma'].iloc[idx] if 'funding_rate_ma' in df.columns else 0
            funding_cumsum = df['funding_rate_cumsum'].iloc[idx] if 'funding_rate_cumsum' in df.columns else 0
            
            # Open Interest (3) - 从合并后的列读取
            oi_normalized = df['oi_normalized'].iloc[idx] if 'oi_normalized' in df.columns else 0
            oi_change = df['oi_change'].iloc[idx] if 'oi_change' in df.columns else 0
            oi_zscore = df['oi_zscore'].iloc[idx] if 'oi_zscore' in df.columns else 0
            
            # 期货溢价 (2)
            futures_premium = df['futures_premium'].iloc[idx] if 'futures_premium' in df.columns else 0
            basis = df['basis'].iloc[idx] if 'basis' in df.columns else 0
            
            # 技术指标 (4)
            delta = df['close'].diff().iloc[idx-14:idx]
            gain = delta.where(delta > 0, 0).mean()
            loss = (-delta.where(delta < 0, 0)).mean()
            rsi = (100 - 100 / (1 + gain / (loss + 1e-10))) / 100 - 0.5 if loss > 0 else 0
            
            bb_ma = df['close'].iloc[idx-20:idx].mean()
            bb_std = df['close'].iloc[idx-20:idx].std()
            bb_position = (df['close'].iloc[idx] - bb_ma) / (2 * bb_std + 1e-10)
            
            ma_7 = df['close'].iloc[idx-168:idx].mean()
            ma_30 = df['close'].iloc[idx-720:idx].mean() if idx >= 720 else ma_7
            trend_strength = (ma_7 - ma_30) / (ma_30 + 1e-10)
            
            ema_12 = df['close'].iloc[max(0,idx-12):idx].ewm(span=12).mean().iloc[-1] if idx >= 12 else df['close'].iloc[idx]
            ema_26 = df['close'].iloc[max(0,idx-26):idx].ewm(span=26).mean().iloc[-1] if idx >= 26 else df['close'].iloc[idx]
            macd = (ema_12 - ema_26) / df['close'].iloc[idx]
            
            # 成交量 (3)
            if 'volume' in df.columns and df['volume'].iloc[idx] > 0:
                vol_ma = df['volume'].iloc[idx-24:idx].mean()
                vol_zscore = (df['volume'].iloc[idx] - vol_ma) / (df['volume'].iloc[idx-168:idx].std() + 1e-10) if idx >= 168 else 0
                vol_trend = df['volume'].iloc[idx-24:idx].mean() / (df['volume'].iloc[idx-168:idx].mean() + 1e-10) - 1 if idx >= 168 else 0
                vol_spike = df['volume'].iloc[idx] / (vol_ma + 1e-10)
            else:
                vol_zscore = vol_trend = vol_spike = 0
            
            features = np.array([
                np.clip(ret_24h, -0.5, 0.5),
                np.clip(ret_7d, -1, 1),
                np.clip(ret_30d, -2, 2),
                np.clip(mom_24h, -0.5, 0.5),
                np.clip(mom_7d, -1, 1),
                mom_consistency,
                np.clip(vol_7d, 0, 3),
                np.clip(vol_30d, 0, 3),
                np.clip(vol_ratio, 0, 5),
                np.clip(vol_of_vol, 0, 1),
                np.clip(funding_rate * 100, -1, 1),
                np.clip(funding_ma * 100, -1, 1),
                np.clip(funding_cumsum * 10, -5, 5),
                np.clip(oi_normalized, 0, 2),
                np.clip(oi_change, -1, 1),
                np.clip(oi_zscore, -5, 5),
                np.clip(futures_premium * 100, -10, 10),
                np.clip(basis * 100, -10, 10),
                np.clip(rsi, -0.5, 0.5),
                np.clip(bb_position, -3, 3),
                np.clip(trend_strength, -0.5, 0.5),
                np.clip(macd, -0.1, 0.1),
                np.clip(vol_zscore, -5, 5),
                np.clip(vol_trend, -2, 2),
                np.clip(vol_spike, 0, 10),
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)
    
    def extract_batch(self, df: pd.DataFrame, start_idx: int = 720) -> np.ndarray:
        """批量提取"""
        return np.array([self.extract(df, i) for i in range(start_idx, len(df))], dtype=np.float32)


class EnhancedRegimeGMM:
    """增强版 6-Regime GMM"""
    
    def __init__(self, n_components: int = 6, random_state: int = 42):
        self.n_components = n_components
        self.feature_extractor = GMMFeatureExtractor()
        
        if SKLEARN_AVAILABLE:
            self.gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                                       n_init=10, max_iter=200, random_state=random_state)
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=min(15, self.feature_extractor.FEATURE_DIM))
        
        self.cluster_to_regime: Dict[int, int] = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'EnhancedRegimeGMM':
        """训练 GMM"""
        if not SKLEARN_AVAILABLE:
            return self
        
        logger.info("提取 GMM v2 特征...")
        X = self.feature_extractor.extract_batch(df)
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        logger.info(f"样本: {len(X)}, 特征: {X.shape[1]}")
        
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        logger.info("训练 6-Regime GMM...")
        self.gmm.fit(X_pca)
        
        labels = self.gmm.predict(X_pca)
        
        # 自适应阈值映射
        mom_vals = X[:, 0]
        vol_vals = X[:, 8]
        funding_vals = X[:, 10]
        
        mom_p20, mom_p40, mom_p60, mom_p80 = np.percentile(mom_vals, [20, 40, 60, 80])
        vol_p50, vol_p75 = np.percentile(vol_vals, [50, 75])
        funding_p25, funding_p75 = np.percentile(funding_vals, [25, 75])
        
        for k in range(self.n_components):
            mask = labels == k
            if mask.sum() > 0:
                feat = X[mask].mean(axis=0)
                momentum, vol_ratio, funding = feat[0], feat[8], feat[10]
                
                if momentum < mom_p20 and (vol_ratio > vol_p75 or funding > funding_p75):
                    self.cluster_to_regime[k] = Regime.STRONG_BEAR
                elif momentum < mom_p40:
                    self.cluster_to_regime[k] = Regime.MODERATE_BEAR
                elif momentum > mom_p80 and funding < funding_p25:
                    self.cluster_to_regime[k] = Regime.STRONG_BULL
                elif momentum > mom_p60:
                    self.cluster_to_regime[k] = Regime.MODERATE_BULL
                elif vol_ratio > vol_p75:
                    self.cluster_to_regime[k] = Regime.VOLATILE_CHOP
                else:
                    self.cluster_to_regime[k] = Regime.NEUTRAL
        
        self.is_fitted = True
        self._print_stats(labels)
        return self
    
    def _print_stats(self, labels):
        """打印 Regime 分布"""
        logger.info("\n📊 6-Regime 分布:")
        counts = {i: 0 for i in range(6)}
        for k in range(self.n_components):
            counts[self.cluster_to_regime.get(k, 2)] += (labels == k).sum()
        for rid, cnt in counts.items():
            logger.info(f"  {Regime.to_name(rid)}: {cnt} ({cnt/len(labels)*100:.1f}%)")
    
    def predict(self, df: pd.DataFrame, idx: int = -1) -> RegimeState:
        """预测 Regime"""
        if not self.is_fitted:
            return RegimeState(2, 'NEUTRAL', np.array([0.1,0.1,0.3,0.1,0.2,0.2]), 0.3, np.zeros(25))
        
        if idx < 0:
            idx = len(df) - 1
        
        features = self.feature_extractor.extract(df, idx)
        
        try:
            X = self.scaler.transform(features.reshape(1, -1))
            X = self.pca.transform(X)
            
            cluster = self.gmm.predict(X)[0]
            probs = self.gmm.predict_proba(X)[0]
            
            regime_probs = np.zeros(6)
            for k, p in enumerate(probs):
                regime_probs[self.cluster_to_regime.get(k, 2)] += p
            regime_probs /= regime_probs.sum() + 1e-10
            
            regime_id = regime_probs.argmax()
            return RegimeState(int(regime_id), Regime.to_name(regime_id),
                              regime_probs.astype(np.float32), float(regime_probs[regime_id]), features)
        except:
            return RegimeState(2, 'NEUTRAL', np.array([0.1,0.1,0.3,0.1,0.2,0.2]), 0.3, features)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'gmm': self.gmm, 'scaler': self.scaler, 'pca': self.pca,
                        'cluster_to_regime': self.cluster_to_regime, 'is_fitted': self.is_fitted}, f)
        logger.info(f"💾 GMM 保存: {path}")
    
    def load(self, path: str) -> 'EnhancedRegimeGMM':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.gmm = data['gmm']
        self.scaler = data['scaler']
        self.pca = data['pca']
        self.cluster_to_regime = data['cluster_to_regime']
        self.is_fitted = data['is_fitted']
        return self


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--save-dir', type=str, default='./models/gmm')
    args = parser.parse_args()
    
    logger.info(f"加载: {args.data}")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    
    gmm = EnhancedRegimeGMM()
    gmm.fit(df)
    
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    gmm.save(f"{args.save_dir}/gmm_v2.pkl")
    
    logger.info("ON 完成!")


if __name__ == "__main__":
    main()

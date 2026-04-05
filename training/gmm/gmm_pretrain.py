"""
GMM 预训练模块 - 6-Regime 检测与标签生成

匹配 HMATS 原系统:
- STRONG_BEAR (0)
- MODERATE_BEAR (1)  
- NEUTRAL (2)
- VOLATILE_CHOP (3)
- MODERATE_BULL (4)
- STRONG_BULL (5)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("GMM")


class Regime:
    """6-Regime 枚举 (匹配 HMATS)"""
    STRONG_BEAR = 0
    MODERATE_BEAR = 1
    NEUTRAL = 2
    VOLATILE_CHOP = 3
    MODERATE_BULL = 4
    STRONG_BULL = 5
    
    NAMES = ['STRONG_BEAR', 'MODERATE_BEAR', 'NEUTRAL', 
             'VOLATILE_CHOP', 'MODERATE_BULL', 'STRONG_BULL']
    
    # 方向偏好
    DIRECTION_BIAS = {
        0: -0.25,  # STRONG_BEAR: 强做空
        1: -0.10,  # MODERATE_BEAR: 温和做空
        2: 0.00,   # NEUTRAL: 中性
        3: 0.00,   # VOLATILE_CHOP: 中性
        4: +0.05,  # MODERATE_BULL: 温和做多
        5: +0.15,  # STRONG_BULL: 做多
    }
    
    @classmethod
    def to_name(cls, idx: int) -> str:
        return cls.NAMES[idx] if 0 <= idx < len(cls.NAMES) else 'UNKNOWN'
    
    @classmethod
    def get_bias(cls, idx: int) -> float:
        return cls.DIRECTION_BIAS.get(idx, 0.0)


@dataclass
class RegimeState:
    """Regime 状态"""
    regime_id: int
    regime_name: str
    probabilities: np.ndarray
    confidence: float
    features: np.ndarray


class GMM_FeatureExtractor:
    """GMM 特征提取器"""
    
    def extract(self, df: pd.DataFrame, idx: int = -1) -> np.ndarray:
        """提取 GMM 特征 (20维)"""
        if idx == -1:
            idx = len(df) - 1
        if idx < 200:
            return np.zeros(20, dtype=np.float32)
        
        try:
            # 动量
            mom_4h = df['close'].iloc[idx] / df['close'].iloc[idx-4] - 1
            mom_24h = df['close'].iloc[idx] / df['close'].iloc[idx-24] - 1
            mom_72h = df['close'].iloc[idx] / df['close'].iloc[idx-72] - 1
            mom_168h = df['close'].iloc[idx] / df['close'].iloc[idx-168] - 1
            mom_accel = mom_4h - (df['close'].iloc[idx-4] / df['close'].iloc[idx-8] - 1)
            mom_consist = 1 if np.sign(mom_4h) == np.sign(mom_24h) == np.sign(mom_72h) else 0
            
            # 波动率
            returns = df['close'].pct_change()
            vol_24h = returns.iloc[idx-24:idx].std() * np.sqrt(24*365)
            vol_168h = returns.iloc[idx-168:idx].std() * np.sqrt(24*365)
            vol_ratio = vol_24h / (vol_168h + 1e-10)
            vol_trend = vol_24h - returns.iloc[idx-48:idx-24].std() * np.sqrt(24*365)
            vol_ma = returns.iloc[max(0,idx-720):idx].std() * np.sqrt(24*365)
            vol_zscore = (vol_24h - vol_ma) / (vol_ma + 1e-10)
            
            # 趋势
            ma_7 = df['close'].iloc[idx-7:idx].mean()
            ma_25 = df['close'].iloc[idx-25:idx].mean()
            ma_50 = df['close'].iloc[idx-50:idx].mean()
            trend_str = (ma_7 - ma_50) / (ma_50 + 1e-10)
            trend_dir = 1 if ma_7 > ma_25 > ma_50 else (-1 if ma_7 < ma_25 < ma_50 else 0)
            ma_align = 1 if ma_7 > ma_25 else -1
            high_20 = df['high'].iloc[idx-20:idx].max()
            low_20 = df['low'].iloc[idx-20:idx].min()
            breakout = (df['close'].iloc[idx] - low_20) / (high_20 - low_20 + 1e-10) - 0.5
            
            # 成交量
            if 'volume' in df.columns:
                vol_24 = df['volume'].iloc[idx-24:idx].mean()
                vol_168 = df['volume'].iloc[idx-168:idx].mean()
                vol_rat = vol_24 / (vol_168 + 1e-10) - 1
                vol_trnd = (vol_24 - df['volume'].iloc[idx-48:idx-24].mean()) / (vol_168 + 1e-10)
                vol_corr = returns.iloc[idx-24:idx].corr(df['volume'].pct_change().iloc[idx-24:idx])
                vol_corr = 0 if np.isnan(vol_corr) else vol_corr
            else:
                vol_rat, vol_trnd, vol_corr = 0, 0, 0
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).iloc[idx-14:idx].mean()
            loss = (-delta.where(delta < 0, 0)).iloc[idx-14:idx].mean()
            rsi = (100 - 100 / (1 + gain/(loss+1e-10))) / 100 - 0.5
            
            # Bollinger
            bb_ma = df['close'].iloc[idx-20:idx].mean()
            bb_std = df['close'].iloc[idx-20:idx].std()
            bb_pos = (df['close'].iloc[idx] - bb_ma) / (2 * bb_std + 1e-10)
            
            # MACD
            ema_12 = df['close'].iloc[idx-12:idx].ewm(span=12).mean().iloc[-1]
            ema_26 = df['close'].iloc[idx-26:idx].ewm(span=26).mean().iloc[-1]
            macd_sig = 1 if (ema_12 - ema_26) > 0 else -1
            
            return np.array([
                np.clip(mom_4h, -0.3, 0.3), np.clip(mom_24h, -0.5, 0.5),
                np.clip(mom_72h, -0.7, 0.7), np.clip(mom_168h, -1, 1),
                np.clip(mom_accel, -0.1, 0.1), mom_consist,
                np.clip(vol_24h, 0, 2), np.clip(vol_ratio, 0.3, 3),
                np.clip(vol_trend, -0.5, 0.5), np.clip(vol_zscore, -3, 3),
                np.clip(trend_str, -0.3, 0.3), trend_dir, ma_align,
                np.clip(breakout, -1, 1), np.clip(vol_rat, -1, 2),
                np.clip(vol_trnd, -1, 1), np.clip(vol_corr, -1, 1),
                np.clip(rsi, -0.5, 0.5), np.clip(bb_pos, -1.5, 1.5), macd_sig,
            ], dtype=np.float32)
        except:
            return np.zeros(20, dtype=np.float32)
    
    def extract_batch(self, df: pd.DataFrame, start_idx: int = 200) -> np.ndarray:
        return np.array([self.extract(df, i) for i in range(start_idx, len(df))])


class RegimeGMM:
    """GMM 6-Regime 检测模型"""
    
    def __init__(self, n_components: int = 6, random_state: int = 42):
        self.n_components = n_components
        self.feature_extractor = GMM_FeatureExtractor()
        
        if SKLEARN_AVAILABLE:
            self.gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                                       n_init=10, max_iter=200, random_state=random_state)
            self.scaler = StandardScaler()
            self.pca = PCA(n_components=min(12, 20))
        else:
            self.gmm = self.scaler = self.pca = None
        
        self.cluster_to_regime: Dict[int, int] = {}
        self.is_fitted = False
        self._use_pca = True
    
    def fit(self, df: pd.DataFrame) -> 'RegimeGMM':
        """训练 GMM"""
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn 不可用")
            return self
        
        logger.info("提取 GMM 特征...")
        X = self.feature_extractor.extract_batch(df)
        logger.info(f"样本: {len(X)}, 特征: {X.shape[1]}")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = self.pca.fit_transform(X_scaled)
        
        logger.info("训练 6-Regime GMM...")
        self.gmm.fit(X_scaled)
        
        # 映射 Cluster -> 6 Regime
        labels = self.gmm.predict(X_scaled)
        
        # 🆕 v4: 使用自适应阈值（基于数据分位数）
        momentum_vals = X[:, 1]  # momentum_24h
        vol_ratio_vals = X[:, 7]  # volatility_ratio
        
        # 计算分位数阈值
        mom_p20, mom_p40, mom_p60, mom_p80 = np.percentile(momentum_vals, [20, 40, 60, 80])
        vol_p50, vol_p75 = np.percentile(vol_ratio_vals, [50, 75])
        
        logger.info(f"📊 自适应阈值: momentum=[{mom_p20:.3f}, {mom_p40:.3f}, {mom_p60:.3f}, {mom_p80:.3f}]")
        logger.info(f"             vol_ratio=[{vol_p50:.3f}, {vol_p75:.3f}]")
        
        for k in range(self.n_components):
            mask = labels == k
            if mask.sum() > 0:
                feat = X[mask].mean(axis=0)
                momentum = feat[1]  # momentum_24h
                vol_ratio = feat[7]  # volatility_ratio
                
                # 🆕 v4: 自适应 6-Regime 映射逻辑
                if momentum < mom_p20 and vol_ratio > vol_p75:
                    self.cluster_to_regime[k] = Regime.STRONG_BEAR
                elif momentum < mom_p40:
                    self.cluster_to_regime[k] = Regime.MODERATE_BEAR
                elif momentum > mom_p80 and vol_ratio < vol_p50:
                    self.cluster_to_regime[k] = Regime.STRONG_BULL
                elif momentum > mom_p60:
                    self.cluster_to_regime[k] = Regime.MODERATE_BULL
                elif vol_ratio > vol_p75:
                    self.cluster_to_regime[k] = Regime.VOLATILE_CHOP
                else:
                    self.cluster_to_regime[k] = Regime.NEUTRAL
        
        self.is_fitted = True
        logger.info("6-Regime GMM 训练完成")
        self._print_regime_stats(X, X_scaled, labels)
        return self
    
    def _print_regime_stats(self, X_raw, X_scaled, labels):
        """打印 Regime 统计"""
        logger.info("\n📊 6-Regime 分布:")
        regime_counts = {i: 0 for i in range(6)}
        for k in range(self.n_components):
            regime_id = self.cluster_to_regime.get(k, Regime.NEUTRAL)
            regime_counts[regime_id] += (labels == k).sum()
        
        for regime_id, count in regime_counts.items():
            pct = count / len(labels) * 100
            logger.info(f"  {Regime.to_name(regime_id)}: {count} ({pct:.1f}%)")
    
    def predict(self, df: pd.DataFrame, idx: int = -1) -> RegimeState:
        """预测 Regime"""
        if not self.is_fitted:
            return RegimeState(Regime.NEUTRAL, 'NEUTRAL', np.array([0.1,0.1,0.2,0.2,0.2,0.2]), 0.2, np.zeros(20))
        
        features = self.feature_extractor.extract(df, idx)
        
        # 🆕 v4: 检查特征维度是否与 scaler 匹配
        expected_features = self.scaler.n_features_in_
        if len(features) != expected_features:
            # 维度不匹配，使用简化预测（基于价格动量）
            if idx < 0:
                idx = len(df) - 1
            if idx < 24:
                return RegimeState(Regime.NEUTRAL, 'NEUTRAL', np.array([0.1,0.1,0.3,0.1,0.2,0.1]), 0.3, features)
            
            # 简单基于价格变化的 regime 判断
            ret_24h = df['close'].iloc[idx] / df['close'].iloc[idx-24] - 1
            vol_24h = df['close'].iloc[idx-24:idx].pct_change().std() * np.sqrt(24*365)
            
            regime_probs = np.array([0.1, 0.15, 0.25, 0.1, 0.2, 0.2])  # 默认分布
            
            if ret_24h < -0.05:
                regime_id = Regime.STRONG_BEAR
                regime_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
            elif ret_24h < -0.02:
                regime_id = Regime.MODERATE_BEAR
                regime_probs = np.array([0.15, 0.45, 0.2, 0.1, 0.05, 0.05])
            elif ret_24h > 0.05:
                regime_id = Regime.STRONG_BULL
                regime_probs = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.5])
            elif ret_24h > 0.02:
                regime_id = Regime.MODERATE_BULL
                regime_probs = np.array([0.05, 0.05, 0.15, 0.1, 0.45, 0.2])
            elif vol_24h > 0.8:
                regime_id = Regime.VOLATILE_CHOP
                regime_probs = np.array([0.1, 0.15, 0.15, 0.4, 0.1, 0.1])
            else:
                regime_id = Regime.NEUTRAL
                regime_probs = np.array([0.1, 0.15, 0.4, 0.15, 0.1, 0.1])
            
            return RegimeState(int(regime_id), Regime.to_name(regime_id), 
                              regime_probs.astype(np.float32), float(regime_probs[regime_id]), features)
        
        # 🆕 v4: 检查 scaler 期望的特征维度
        expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 20
        
        if len(features) != expected_features:
            # 维度不匹配，使用规则回退
            logger.warning(f"特征维度不匹配: 提取={len(features)}, 期望={expected_features}，使用规则回退")
            
            ret_24h = df['close'].iloc[idx] / df['close'].iloc[max(0, idx-24)] - 1 if idx >= 24 else 0
            vol_24h = df['close'].pct_change().iloc[max(0,idx-24):idx].std() * np.sqrt(24*365) if idx >= 24 else 0.5
            
            if ret_24h < -0.05:
                regime_id = Regime.STRONG_BEAR
                regime_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
            elif ret_24h < -0.02:
                regime_id = Regime.MODERATE_BEAR
                regime_probs = np.array([0.15, 0.45, 0.2, 0.1, 0.05, 0.05])
            elif ret_24h > 0.05:
                regime_id = Regime.STRONG_BULL
                regime_probs = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.5])
            elif ret_24h > 0.02:
                regime_id = Regime.MODERATE_BULL
                regime_probs = np.array([0.05, 0.05, 0.15, 0.1, 0.45, 0.2])
            elif vol_24h > 0.8:
                regime_id = Regime.VOLATILE_CHOP
                regime_probs = np.array([0.1, 0.15, 0.15, 0.4, 0.1, 0.1])
            else:
                regime_id = Regime.NEUTRAL
                regime_probs = np.array([0.1, 0.15, 0.4, 0.15, 0.1, 0.1])
            
            return RegimeState(int(regime_id), Regime.to_name(regime_id), 
                              regime_probs.astype(np.float32), float(regime_probs[regime_id]), features)
        
        X = self.scaler.transform(features.reshape(1, -1))
        X = self.pca.transform(X)
        
        cluster = self.gmm.predict(X)[0]
        probs = self.gmm.predict_proba(X)[0]
        
        # 聚合到 6 Regime 概率
        regime_probs = np.zeros(6)
        for k, p in enumerate(probs):
            regime_probs[self.cluster_to_regime.get(k, Regime.NEUTRAL)] += p
        regime_probs /= regime_probs.sum() + 1e-10
        
        regime_id = regime_probs.argmax()
        return RegimeState(int(regime_id), Regime.to_name(regime_id), 
                          regime_probs.astype(np.float32), float(regime_probs[regime_id]), features)
    
    def predict_batch(self, df: pd.DataFrame, start_idx: int = 200) -> List[RegimeState]:
        return [self.predict(df, i) for i in range(start_idx, len(df))]
    
    def generate_labels(self, df: pd.DataFrame, start_idx: int = 200) -> pd.DataFrame:
        """生成 6-Regime 标签"""
        states = self.predict_batch(df, start_idx)
        return pd.DataFrame([{
            'idx': start_idx + i, 'regime_id': s.regime_id, 'regime_name': s.regime_name,
            'confidence': s.confidence, 
            'prob_strong_bear': s.probabilities[0],
            'prob_moderate_bear': s.probabilities[1],
            'prob_neutral': s.probabilities[2],
            'prob_volatile': s.probabilities[3],
            'prob_moderate_bull': s.probabilities[4],
            'prob_strong_bull': s.probabilities[5],
        } for i, s in enumerate(states)])
    
    def fit_from_features(self, features_df: pd.DataFrame) -> 'RegimeGMM':
        """从已有的 GMM 特征 DataFrame 训练 (匹配你的数据格式)"""
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn 不可用")
            return self
        
        logger.info("从预计算特征训练 GMM...")
        
        # 你的数据格式
        feature_cols = [
            'return_1h', 'return_4h', 'return_24h', 'return_7d',
            'volatility_24h', 'volatility_7d', 'vol_of_vol', 'vol_percentile',
            'momentum_4h', 'momentum_24h', 'momentum_consistency',
            'zscore_24h', 'zscore_168h', 'volume_zscore', 'volume_trend'
        ]
        
        # 检查可用列
        available_cols = [c for c in feature_cols if c in features_df.columns]
        logger.info(f"使用特征: {available_cols}")
        
        X = features_df[available_cols].fillna(0).values
        logger.info(f"样本: {len(X)}, 特征: {X.shape[1]}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        n_pca = min(10, X.shape[1])
        self.pca = PCA(n_components=n_pca)
        X_scaled = self.pca.fit_transform(X_scaled)
        
        logger.info("训练 6-Regime GMM...")
        self.gmm.fit(X_scaled)
        
        # 映射
        labels = self.gmm.predict(X_scaled)
        for k in range(self.n_components):
            mask = labels == k
            if mask.sum() > 0:
                feat = X[mask].mean(axis=0)
                
                # 根据你的特征列索引
                ret_24h = feat[2] if len(feat) > 2 else 0  # return_24h
                vol_24h = feat[4] if len(feat) > 4 else 0  # volatility_24h
                mom_24h = feat[9] if len(feat) > 9 else 0  # momentum_24h
                
                # 6-Regime 映射
                if ret_24h < -0.05 or mom_24h < -0.08:
                    self.cluster_to_regime[k] = Regime.STRONG_BEAR
                elif ret_24h < -0.02 or mom_24h < -0.03:
                    self.cluster_to_regime[k] = Regime.MODERATE_BEAR
                elif ret_24h > 0.05 or mom_24h > 0.08:
                    self.cluster_to_regime[k] = Regime.STRONG_BULL
                elif ret_24h > 0.02 or mom_24h > 0.03:
                    self.cluster_to_regime[k] = Regime.MODERATE_BULL
                elif vol_24h > 0.8:  # 高波动
                    self.cluster_to_regime[k] = Regime.VOLATILE_CHOP
                else:
                    self.cluster_to_regime[k] = Regime.NEUTRAL
        
        self.is_fitted = True
        self._use_pca = True
        logger.info("6-Regime GMM 训练完成")
        self._print_regime_stats(X, X_scaled, labels)
        return self
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'gmm': self.gmm, 'scaler': self.scaler, 'pca': self.pca,
                        'cluster_to_regime': self.cluster_to_regime, 'n_components': self.n_components}, f)
        logger.info(f"GMM 已保存: {path}")
    
    def load(self, path: str) -> 'RegimeGMM':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.gmm, self.scaler, self.pca = data['gmm'], data['scaler'], data['pca']
        self.cluster_to_regime, self.n_components = data['cluster_to_regime'], data['n_components']
        self.is_fitted = True
        logger.info(f"GMM 已加载: {path}")
        return self


class GMMEnhancedLabeler:
    """GMM 增强标签生成器 (6 Regime)"""
    
    def __init__(self, gmm: RegimeGMM):
        self.gmm = gmm
    
    def generate_direction_labels(self, df: pd.DataFrame, lookahead: int = 4, 
                                   threshold: float = 0.01) -> pd.DataFrame:
        """生成方向标签"""
        regime_labels = self.gmm.generate_labels(df)
        df = df.copy()
        df['future_ret'] = df['close'].shift(-lookahead) / df['close'] - 1
        
        labels = []
        for _, row in regime_labels.iterrows():
            idx = row['idx']
            if idx >= len(df) - lookahead:
                continue
            
            fut_ret = df['future_ret'].iloc[idx]
            direction = 0 if fut_ret < -threshold else (2 if fut_ret > threshold else 1)
            
            # 6 Regime 方向匹配
            regime_id = row['regime_id']
            regime_match = (
                (regime_id in [Regime.STRONG_BEAR, Regime.MODERATE_BEAR] and direction == 0) or
                (regime_id in [Regime.STRONG_BULL, Regime.MODERATE_BULL] and direction == 2)
            )
            
            labels.append({
                'idx': idx, 'direction': direction, 'future_return': fut_ret,
                'regime_id': regime_id,
                'regime_probs': np.array([
                    row['prob_strong_bear'], row['prob_moderate_bear'],
                    row['prob_neutral'], row['prob_volatile'],
                    row['prob_moderate_bull'], row['prob_strong_bull']
                ]),
                'regime_match': regime_match,
                'label_confidence': row['confidence'] if regime_match else 0.5,
                'direction_bias': Regime.get_bias(regime_id),
            })
        
        return pd.DataFrame(labels)
    
    def generate_rtg_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成 RTG 标签"""
        regime_labels = self.gmm.generate_labels(df)
        rtg_scales = [1, 4, 12, 24, 48, 96, 168]
        
        labels = []
        for _, row in regime_labels.iterrows():
            idx = row['idx']
            rtg_values = []
            for scale in rtg_scales:
                if idx + scale < len(df):
                    rtg_values.append(df['close'].iloc[idx + scale] / df['close'].iloc[idx] - 1)
                else:
                    rtg_values.append(0)
            
            labels.append({
                'idx': idx, **{f'rtg_{s}': v for s, v in zip(rtg_scales, rtg_values)},
                'regime_id': row['regime_id'],
                'regime_probs': np.array([
                    row['prob_strong_bear'], row['prob_moderate_bear'],
                    row['prob_neutral'], row['prob_volatile'],
                    row['prob_moderate_bull'], row['prob_strong_bull']
                ]),
            })
        
        return pd.DataFrame(labels)

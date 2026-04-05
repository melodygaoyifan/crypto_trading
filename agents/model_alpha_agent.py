"""
================================================================================
HMATS v6 - Model Alpha Agent
================================================================================

Advisory agent that provides model-based directional signals to the v6
authority fusion pipeline.  Outputs are ADVISORY ONLY; sizing, risk, and
execution are handled elsewhere.

Role in v6:
  - Consumes a UnifiedState built from runtime market_data / agent_signals.
  - Produces ModelIntent objects (direction, confidence, horizon).
  - Provides to_agent_signals() for integration with authority_fusion.
  - Respects governance: can be vetoed / weight-capped by risk layer.

Constraints (v6 non-negotiable):
  - No embedded execution (no stops/TPs, no position management).
  - Freshness/completeness gated: missing/stale inputs -> neutral + diagnostics.
  - Weight controllers are optional and safe (no hard-fail if absent).
================================================================================
"""

import logging
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import json
from pathlib import Path

# PyTorch for Decision Transformer
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    # Create dummy nn module for type hints
    class nn:
        class Module:
            pass
        class Sequential:
            pass
        class Linear:
            pass
        class LayerNorm:
            pass
        class GELU:
            pass
        class Dropout:
            pass

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ModelType(Enum):
    """模型类型"""
    AUTO = "auto"
    DECISION_TRANSFORMER = "decision_transformer"
    SEQUENCE_ALPHA = "sequence_alpha"
    REGIME_CLASSIFIER = "regime_classifier"
    CUSTOM = "custom"


class IntentDirection(Enum):
    """意图方向"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class ModelIntent:
    """
    模型 Alpha 输出的交易意图
    
    这是模型对市场的判断，需要通过 authority_fusion 进入决策链
    """
    # 核心属性
    asset: str
    direction: IntentDirection
    confidence: float  # 0-1
    horizon: str  # "short", "medium", "long"
    
    # 时间戳
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 解释
    explanation_tokens: List[str] = field(default_factory=list)
    
    # 模型来源
    model_type: ModelType = ModelType.DECISION_TRANSFORMER
    model_version: str = "v1.0"
    
    # 输入状态摘要
    state_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    raw_prediction: float = 0.0  # 原始预测值 (-1 to 1)
    features_used: List[str] = field(default_factory=list)
    effective_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "timestamp": self.timestamp.isoformat(),
            "explanation_tokens": self.explanation_tokens,
            "model_type": self.model_type.value,
            "model_version": self.model_version,
            "raw_prediction": self.raw_prediction,
            "effective_weight": self.effective_weight,
        }

    def to_agent_signals(self) -> Dict[str, Any]:
        """
        Convert to v6 agent_signals-compatible flat dict.

        Keys are namespaced ``model_alpha_*`` so they never collide with
        quant_direction / quant_confidence.
        """
        dir_map = {
            IntentDirection.LONG: 1.0,
            IntentDirection.SHORT: -1.0,
            IntentDirection.NEUTRAL: 0.0,
        }
        return {
            "model_alpha_direction": dir_map.get(self.direction, 0.0),
            "model_alpha_confidence": round(self.confidence, 4),
            "model_alpha_raw_prediction": round(self.raw_prediction, 6),
            "model_alpha_horizon": self.horizon,
            "model_alpha_weight": round(self.effective_weight, 4),
            "model_alpha_asof_timestamp": self.timestamp.isoformat(),
            "model_alpha_data_quality": 1.0,
            "model_alpha_reasons": self.explanation_tokens[:10],
            "model_alpha_model_type": self.model_type.value,
            "model_alpha_model_version": self.model_version,
        }


@dataclass
class UnifiedState:
    """统一输入状态"""
    timestamp: datetime
    
    # 价格数据
    prices: Dict[str, float] = field(default_factory=dict)
    returns_1h: Dict[str, float] = field(default_factory=dict)
    returns_24h: Dict[str, float] = field(default_factory=dict)
    
    # 波动率
    volatility: Dict[str, float] = field(default_factory=dict)
    volatility_regime: str = "normal"
    
    # 订单簿
    lob_imbalance: Dict[str, float] = field(default_factory=dict)
    spread_bps: Dict[str, float] = field(default_factory=dict)
    liquidity_regime: str = "normal"
    
    # 链上
    whale_flow: Dict[str, float] = field(default_factory=dict)
    exchange_flow: Dict[str, float] = field(default_factory=dict)
    
    # 情绪
    fear_greed: int = 50
    sentiment_score: float = 0.0
    funding_rate: Dict[str, float] = field(default_factory=dict)
    
    # 仓位
    positions: Dict[str, float] = field(default_factory=dict)
    pnl_unrealized: Dict[str, float] = field(default_factory=dict)
    
    # 宏观
    risk_on_off: float = 0.0
    usd_strength: float = 0.0
    
    # -- coverage tracking (set by from_runtime) ----------------------------
    _coverage: float = 1.0   # fraction of expected fields present [0..1]
    _missing: List[str] = field(default_factory=list)
    _data_age_seconds: float = 0.0
    _obs_126_by_asset: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_runtime(
        cls,
        primary_asset: str,
        market_data: Dict[str, Any],
        agent_signals: Optional[Dict[str, Any]] = None,
        positions: Optional[Dict[str, float]] = None,
        obs_126: Optional[np.ndarray] = None,
    ) -> "UnifiedState":
        """
        Build a UnifiedState from the v6 runtime dicts emitted by main.py.

        Robust to missing fields - records coverage and missing keys for
        downstream freshness gating.
        """
        agent_signals = agent_signals or {}
        positions = positions or {}
        md = market_data or {}

        missing: List[str] = []

        def _get(key, default=None):
            val = md.get(key, default)
            if val is None:
                missing.append(key)
                return default
            return val

        ts = md.get("timestamp")
        if ts is None:
            ts = datetime.now(timezone.utc)
            missing.append("timestamp")
        elif isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)
                missing.append("timestamp_parse")

        # Prices - accept flat "{asset}_price" or nested dicts
        prices: Dict[str, float] = {}
        for asset in ("BTC", "ETH", "SOL"):
            p = md.get(f"{asset.lower()}_price") or md.get(f"price_{asset.lower()}")
            if p is None and isinstance(md.get(asset), dict):
                p = md[asset].get("close")
            if p is not None:
                prices[asset] = float(p)
            else:
                missing.append(f"{asset}_price")

        data_age = float(_get("data_age_seconds", 0.0) or 0.0)

        state = cls(
            timestamp=ts,
            prices=prices,
            returns_1h={a: float(md.get(f"{a.lower()}_returns_1h", 0.0) or 0.0) for a in ("BTC", "ETH", "SOL")},
            returns_24h={a: float(md.get(f"{a.lower()}_returns_24h", 0.0) or 0.0) for a in ("BTC", "ETH", "SOL")},
            volatility={a: float(md.get(f"{a.lower()}_volatility", md.get("volatility_4h", 0.0)) or 0.0) for a in ("BTC", "ETH", "SOL")},
            volatility_regime=str(_get("volatility_regime", "normal")),
            lob_imbalance={primary_asset: float(agent_signals.get("order_book_imbalance", 0.0) or 0.0)},
            spread_bps={primary_asset: float(agent_signals.get("spread_bps", 0.0) or 0.0)},
            fear_greed=int(md.get("fear_greed", md.get("fear_greed_index", 50)) or 50),
            sentiment_score=float(agent_signals.get("sentiment_zscore", 0.0) or 0.0),
            funding_rate={primary_asset: float(md.get("funding_rate", 0.0) or 0.0)},
            positions=positions,
            risk_on_off=float(md.get("risk_on_off", 0.0) or 0.0),
            usd_strength=float(md.get("usd_strength", 0.0) or 0.0),
        )

        # Coverage is evaluated against the single-asset runtime payload used by the
        # live loop, not against a hypothetical cross-asset snapshot. When a full
        # 126-dim runtime observation is already available, let freshness/confidence
        # gates decide usage instead of failing early on missing BTC/ETH/SOL sibling
        # prices.
        expected = 3  # {primary asset price, timestamp, data_age_seconds}
        present = expected
        if primary_asset not in prices:
            present -= 1
        if "timestamp" in missing or "timestamp_parse" in missing:
            present -= 1
        if "data_age_seconds" in missing:
            present -= 1
        state._coverage = max(0.0, present / expected)
        state._missing = missing
        state._data_age_seconds = data_age
        if obs_126 is not None:
            try:
                _obs_arr = np.asarray(obs_126, dtype=np.float32)
                state._obs_126_by_asset[primary_asset] = _obs_arr
                if _obs_arr.size >= 120:
                    state._coverage = max(state._coverage, 1.0)
            except Exception:
                pass
        return state

    def to_feature_vector(self, asset: str) -> np.ndarray:
        """Convert state to model input vector."""
        obs_126 = self._obs_126_by_asset.get(asset)
        if obs_126 is not None:
            return np.asarray(obs_126, dtype=np.float32)

        features = [
            self.returns_1h.get(asset, 0.0),
            self.returns_24h.get(asset, 0.0),
            self.volatility.get(asset, 0.0),
            self.lob_imbalance.get(asset, 0.0),
            self.spread_bps.get(asset, 0.0) / 100,  # 归一化
            self.whale_flow.get(asset, 0.0) / 1e6,  # 归一化到百万
            self.exchange_flow.get(asset, 0.0) / 1e6,
            (self.fear_greed - 50) / 50,  # -1 to 1
            self.sentiment_score,
            self.funding_rate.get(asset, 0.0) * 100,  # 放大
            self.positions.get(asset, 0.0),
            self.risk_on_off,
            self.usd_strength,
        ]
        return np.array(features, dtype=np.float32)


# =============================================================================
# MODEL ALPHA AGENT
# =============================================================================

class ModelAlphaAgent:
    """
    模型 Alpha Agent
    
    职责:
    1. 消费统一状态输入
    2. 调用模型生成预测
    3. 输出 ModelIntent
    4. 接入 authority_fusion
    
    约束:
    - 不直接执行交易
    - 必须通过 governance 检查
    - 权重受风险控制约束
    """
    
    def __init__(
        self,
        model_type: ModelType = ModelType.AUTO,
        model_path: str = "models/decision_transformer",
        assets: List[str] = None,
        allow_mock_model: bool = False,
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.sequence_model_path = str(Path("models") / "model_alpha")
        self.assets = assets or ["BTC", "ETH", "SOL"]
        self.allow_mock_model = allow_mock_model

        # 模型状态
        self._model_loaded = False
        self._model = None
        self._models_by_asset: Dict[str, Any] = {}
        self._model_backend_by_asset: Dict[str, Dict[str, str]] = {}
        self._obs_builder = None

        # 历史记录（用于 PnL 追踪）
        self._intent_history: deque = deque(maxlen=100)
        self._pnl_history: deque = deque(maxlen=100)

        # 性能指标
        self._stats = {
            "total_intents": 0,
            "accepted_intents": 0,
            "vetoed_intents": 0,
            "correct_predictions": 0,
            "rolling_pnl": 0.0,
            "rolling_sharpe": 0.0,
            "consecutive_failures": 0,
        }

        # 配置
        self._confidence_threshold = 0.35  # profitability-first threshold
        self._min_abs_prediction = 0.12
        self._soft_abs_prediction = 0.03
        self._soft_context_max_weight = 0.35
        self._max_weight = 1.0            # 最大权重（可被风控降低）
        self._current_weight = 1.0        # 当前权重

        # 加载模型
        self._load_model()

        logger.info(
            f"ModelAlphaAgent initialized: type={model_type.value}, "
            f"assets={self.assets}, loaded_assets={list(self._models_by_asset.keys())}"
        )
    
    def _load_model(self):
        """Load profitability-first model weights. Fail-closed unless mock explicitly enabled."""
        if TORCH_AVAILABLE:
            for asset in self.assets:
                _loaded = False

                if self.model_type in (ModelType.AUTO, ModelType.SEQUENCE_ALPHA, ModelType.CUSTOM):
                    checkpoint_path = self._resolve_sequence_model_checkpoint(asset)
                    if checkpoint_path is not None:
                        try:
                            logger.info(f"Loading Sequence Alpha model for {asset} from {checkpoint_path}")
                            _seq_model = RealSequenceAlphaModel(model_path=str(checkpoint_path))
                            if _seq_model.model is not None:
                                self._register_loaded_model(
                                    asset=asset,
                                    model=_seq_model,
                                    model_type=ModelType.SEQUENCE_ALPHA,
                                    model_version=f"sequence_alpha_v1:{asset.lower()}",
                                )
                                _loaded = True
                            else:
                                logger.warning(f"Sequence alpha checkpoint for {asset} loaded without weights: {checkpoint_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load sequence alpha for {asset} from {checkpoint_path}: {e}")

                if (not _loaded) and self.model_type in (
                    ModelType.AUTO,
                    ModelType.DECISION_TRANSFORMER,
                    ModelType.CUSTOM,
                ):
                    checkpoint_path = self._resolve_model_checkpoint(asset)
                    if checkpoint_path is not None:
                        try:
                            logger.info(f"Loading Decision Transformer for {asset} from {checkpoint_path}")
                            _dt_model = RealDecisionTransformer(
                                model_path=str(checkpoint_path),
                                state_dim=126,
                                action_dim=1,
                                hidden_dim=384,
                            )
                            if _dt_model.model is not None:
                                self._register_loaded_model(
                                    asset=asset,
                                    model=_dt_model,
                                    model_type=ModelType.DECISION_TRANSFORMER,
                                    model_version=f"dt_v3.2:{asset.lower()}",
                                )
                            else:
                                logger.warning(f"DT checkpoint for {asset} loaded without model weights: {checkpoint_path}")
                        except Exception as e:
                            logger.warning(f"Failed to load DT for {asset} from {checkpoint_path}: {e}")

            if self._models_by_asset:
                self._model_loaded = True
                self._model = next(iter(self._models_by_asset.values()))
                _backend_summary = ", ".join(
                    f"{k}:{v.get('model_type', 'unknown')}"
                    for k, v in sorted(self._model_backend_by_asset.items())
                )
                logger.info(
                    f"Model alpha backends loaded: "
                    f"{_backend_summary}"
                )
                return

        # v6.4: fail-closed - only use mock if explicitly allowed
        if self.allow_mock_model:
            logger.info(f"Using MockDecisionTransformer (allow_mock_model=True, model not at {self.model_path})")
            self._model = MockDecisionTransformer()
            self._models_by_asset = {asset: self._model for asset in self.assets}
            self._model_backend_by_asset = {
                asset: {
                    "model_type": ModelType.DECISION_TRANSFORMER.value,
                    "model_version": f"mock:{asset.lower()}",
                }
                for asset in self.assets
            }
            self._model_loaded = True
        else:
            logger.warning(
                f"Model not found in {self.sequence_model_path} or {self.model_path} and allow_mock_model=False. "
                "Agent will emit neutral signals (direction=0, confidence=0)."
            )
            self._model = None
            self._models_by_asset = {}
            self._model_backend_by_asset = {}
            self._model_loaded = False

    def _register_loaded_model(self, asset: str, model: Any, model_type: ModelType, model_version: str) -> None:
        self._models_by_asset[asset] = model
        self._model_backend_by_asset[asset] = {
            "model_type": model_type.value,
            "model_version": model_version,
        }

    def _resolve_model_checkpoint(self, asset: str) -> Optional[Path]:
        """Resolve the best available Decision Transformer checkpoint for an asset."""
        base = Path(self.model_path)
        candidates: List[Path] = []

        if base.is_file():
            candidates.append(base)
        else:
            asset_dir = base / asset.upper()
            candidates.extend([
                asset_dir / "dt_v32_best.pt",
                asset_dir / "dt_v32_final.pt",
                asset_dir / "model.pt",
                base / f"{asset.upper()}_dt_v32_best.pt",
                base / "model.pt",
            ])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_sequence_model_checkpoint(self, asset: str) -> Optional[Path]:
        base = Path(self.sequence_model_path)
        candidates = [
            base / asset.upper() / "sequence_alpha_v1_best.pt",
            base / asset.upper() / "sequence_alpha_v1_final.pt",
            base / asset.upper() / "model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _get_model(self, asset: str):
        return self._models_by_asset.get(asset) or self._model

    def _get_obs_builder(self):
        """Lazily construct RuntimeObsBuilder so the agent can consume 126-dim runtime obs."""
        if self._obs_builder is not None:
            return self._obs_builder
        try:
            from drl.runtime_obs_builder import RuntimeObsBuilder
            self._obs_builder = RuntimeObsBuilder(
                manifest_path="configs/feature_manifest.json",
                scaler_dir="models/retrained",
                assets=self.assets,
            )
        except Exception as e:
            logger.debug(f"[MODEL_ALPHA] RuntimeObsBuilder unavailable: {e}")
            self._obs_builder = None
        return self._obs_builder
    
    # v6 freshness thresholds
    MAX_DATA_AGE_SECONDS: float = 30.0
    MIN_COVERAGE: float = 0.6

    def generate_intent(self, state: UnifiedState, asset: str) -> Optional[ModelIntent]:
        """
        Generate a model intent for *asset*.

        v6 freshness/completeness gating:
        - data_age_seconds > MAX_DATA_AGE_SECONDS -> None + log
        - coverage < MIN_COVERAGE -> None + log
        - model not loaded / asset unknown -> None

        Returns:
            ModelIntent or None (with diagnostics in ``_last_gating_reason``).
        """
        self._last_gating_reason: Optional[str] = None

        asset_model = self._get_model(asset)
        if not self._model_loaded or asset not in self.assets or asset_model is None:
            self._last_gating_reason = "model_not_loaded" if not self._model_loaded else "unknown_asset"
            return None

        # -- v6 freshness gate ------------------------------------------------
        if hasattr(state, "_data_age_seconds") and state._data_age_seconds > self.MAX_DATA_AGE_SECONDS:
            self._last_gating_reason = f"stale_data_{state._data_age_seconds:.1f}s"
            logger.debug(f"[MODEL_ALPHA] Gated: stale data ({state._data_age_seconds:.1f}s) for {asset}")
            return None

        if hasattr(state, "_coverage") and state._coverage < self.MIN_COVERAGE:
            self._last_gating_reason = f"low_coverage_{state._coverage:.2f}"
            logger.debug(f"[MODEL_ALPHA] Gated: low coverage ({state._coverage:.2f}) for {asset}")
            return None

        try:
            # Convert to feature vector
            features = np.asarray(state.to_feature_vector(asset), dtype=np.float32)
            if hasattr(asset_model, "prepare_features"):
                features = np.asarray(asset_model.prepare_features(features), dtype=np.float32)
            expected_dim = int(getattr(asset_model, "state_dim", len(features)) or len(features))
            if len(features) != expected_dim:
                self._last_gating_reason = f"obs_dim_mismatch_{len(features)}_{expected_dim}"
                logger.debug(
                    f"[MODEL_ALPHA] Gated: obs_dim mismatch for {asset} "
                    f"(got={len(features)}, need={expected_dim})"
                )
                return None
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 模型预测
            raw_pred, confidence = asset_model.predict(features)

            _backend = self._model_backend_by_asset.get(asset, {})
            _effective_type = _backend.get("model_type", self.model_type.value)
            _effective_version = _backend.get("model_version", f"{self.model_type.value}:{asset.lower()}")
            _effective_model_type = next(
                (mt for mt in ModelType if mt.value == _effective_type),
                self.model_type,
            )

            # 应用当前权重调整置信度
            adjusted_confidence = confidence * self._current_weight
            
            # 低置信度过滤
            if adjusted_confidence < self._confidence_threshold:
                self._last_gating_reason = f"low_confidence_{adjusted_confidence:.2f}"
                return None

            effective_weight = 1.0
            if abs(raw_pred) < self._min_abs_prediction:
                if (
                    _effective_type == ModelType.SEQUENCE_ALPHA.value
                    and abs(raw_pred) >= self._soft_abs_prediction
                ):
                    soft_span = max(self._min_abs_prediction - self._soft_abs_prediction, 1e-6)
                    soft_strength = max(
                        0.0,
                        min(1.0, (abs(raw_pred) - self._soft_abs_prediction) / soft_span),
                    )
                    effective_weight = 0.10 + soft_strength * (self._soft_context_max_weight - 0.10)
                    adjusted_confidence *= 0.85
                else:
                    self._last_gating_reason = f"weak_edge_{raw_pred:.3f}"
                    return None
            
            # 确定方向
            if raw_pred > 0.1:
                direction = IntentDirection.LONG
            elif raw_pred < -0.1:
                direction = IntentDirection.SHORT
            else:
                direction = IntentDirection.NEUTRAL
            
            # 确定时间范围
            if abs(raw_pred) > 0.5:
                horizon = "short"  # 强信号 -> 短期
            elif abs(raw_pred) > 0.3:
                horizon = "medium"
            else:
                horizon = "long"
            
            # 生成解释
            explanation = self._generate_explanation(state, asset, raw_pred, features)
            if effective_weight < 0.999:
                explanation.append(f"soft_context_weight:{effective_weight:.2f}")
                explanation.append(f"soft_context_edge:{raw_pred:.3f}")
            
            intent = ModelIntent(
                asset=asset,
                direction=direction,
                confidence=adjusted_confidence,
                horizon=horizon,
                timestamp=datetime.now(),
                explanation_tokens=explanation,
                model_type=_effective_model_type,
                model_version=_effective_version,
                raw_prediction=raw_pred,
                state_summary={
                    "fear_greed": state.fear_greed,
                    "volatility": state.volatility.get(asset, 0),
                    "lob_imbalance": state.lob_imbalance.get(asset, 0),
                    "coverage": getattr(state, "_coverage", 1.0),
                },
                features_used=[
                    "runtime_obs_126" if len(features) == 126 else "legacy_state_13"
                ],
                effective_weight=effective_weight,
            )
            
            # 记录
            self._intent_history.append(intent)
            self._stats["total_intents"] += 1
            
            return intent
            
        except Exception as e:
            logger.error(f"Error generating intent for {asset}: {e}")
            return None
    
    def generate_all_intents(self, state: UnifiedState) -> List[ModelIntent]:
        """为所有资产生成意图"""
        intents = []
        for asset in self.assets:
            intent = self.generate_intent(state, asset)
            if intent and intent.direction != IntentDirection.NEUTRAL:
                intents.append(intent)
        return intents
    
    def _generate_explanation(
        self,
        state: UnifiedState,
        asset: str,
        prediction: float,
        features: np.ndarray,
    ) -> List[str]:
        """生成解释 tokens"""
        tokens = []
        
        # 方向解释
        direction = "bullish" if prediction > 0 else "bearish"
        tokens.append(f"model_direction:{direction}")
        
        # 主要影响因素
        if abs(state.fear_greed - 50) > 20:
            sentiment = "fear" if state.fear_greed < 50 else "greed"
            tokens.append(f"sentiment_extreme:{sentiment}")
        
        if abs(state.lob_imbalance.get(asset, 0)) > 0.3:
            imbalance = "bid_heavy" if state.lob_imbalance.get(asset, 0) > 0 else "ask_heavy"
            tokens.append(f"lob_imbalance:{imbalance}")
        
        vol = state.volatility.get(asset, 0)
        if vol > 1.0:
            tokens.append("high_volatility")
        elif vol < 0.3:
            tokens.append("low_volatility")
        
        whale = state.whale_flow.get(asset, 0)
        if abs(whale) > 100000:
            direction = "accumulating" if whale > 0 else "distributing"
            tokens.append(f"whale_{direction}")
        
        return tokens
    
    # =========================================================================
    # FEEDBACK & LEARNING
    # =========================================================================
    
    def record_outcome(self, intent: ModelIntent, accepted: bool, vetoed_by: str = ""):
        """
        记录意图结果
        
        由 authority_fusion 调用
        """
        if accepted:
            self._stats["accepted_intents"] += 1
            self._stats["consecutive_failures"] = 0
        else:
            self._stats["vetoed_intents"] += 1
            logger.info(f"ModelIntent vetoed: asset={intent.asset}, vetoed_by={vetoed_by}")
    
    def record_pnl(self, asset: str, pnl_pct: float, intent_time: datetime):
        """
        记录 PnL 结果
        
        用于权重自适应
        """
        self._pnl_history.append({
            "asset": asset,
            "pnl_pct": pnl_pct,
            "timestamp": intent_time,
        })
        
        # 更新统计
        if pnl_pct > 0:
            self._stats["correct_predictions"] += 1
            self._stats["consecutive_failures"] = 0
        else:
            self._stats["consecutive_failures"] += 1
        
        # 计算滚动 PnL
        recent_pnl = [p["pnl_pct"] for p in list(self._pnl_history)[-30:]]
        if recent_pnl:
            self._stats["rolling_pnl"] = sum(recent_pnl)
            
            # 计算 Sharpe
            if len(recent_pnl) > 5:
                mean_ret = np.mean(recent_pnl)
                std_ret = np.std(recent_pnl)
                if std_ret > 0:
                    self._stats["rolling_sharpe"] = mean_ret / std_ret * np.sqrt(252)
    
    # =========================================================================
    # WEIGHT CONTROL
    # =========================================================================
    
    def set_max_weight(self, weight: float):
        """
        Set maximum weight (called by correlation/volatility controllers).

        Safe: never hard-fails, clamps to [0, 1], defaults to 1.0 if absent.
        """
        try:
            self._max_weight = max(0.0, min(1.0, float(weight)))
            self._current_weight = min(self._current_weight, self._max_weight)
        except (TypeError, ValueError):
            logger.warning(f"[MODEL_ALPHA] set_max_weight: invalid value {weight!r}, keeping {self._max_weight}")

    def update_weight(self, new_weight: float):
        """
        Update current weight (called by adaptive_strategy_weight_manager).

        Safe: never hard-fails, clamps to [0, max_weight].
        """
        try:
            self._current_weight = max(0.0, min(self._max_weight, float(new_weight)))
        except (TypeError, ValueError):
            logger.warning(f"[MODEL_ALPHA] update_weight: invalid value {new_weight!r}, keeping {self._current_weight}")
    
    def get_weight(self) -> float:
        """获取当前权重"""
        return self._current_weight
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats["total_intents"]
        accepted = self._stats["accepted_intents"]
        
        return {
            **self._stats,
            "acceptance_rate": accepted / total if total > 0 else 0,
            "current_weight": self._current_weight,
            "max_weight": self._max_weight,
            "model_loaded": self._model_loaded,
            "model_type": self.model_type.value,
            "loaded_assets": sorted(self._models_by_asset.keys()),
            "loaded_backends": dict(self._model_backend_by_asset),
        }
    
    def get_for_authority_fusion(self) -> Dict[str, Any]:
        """Return authority_fusion metadata."""
        return {
            "source_name": "model_alpha",
            "weight": self._current_weight,
            "confidence_threshold": self._confidence_threshold,
            "stats": self.get_stats(),
        }

    def to_agent_signals(
        self,
        asset: str,
        state: Optional[UnifiedState] = None,
        market_data: Optional[Dict] = None,
        agent_signals: Optional[Dict] = None,
        positions: Optional[Dict] = None,
        ohlcv_df: Optional[Any] = None,
        gmm_probs: Optional[List[float]] = None,
        env_state: Optional[Dict[str, Any]] = None,
        obs_builder: Optional[Any] = None,
        obs_126: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Convenience: generate intent and convert to v6 agent_signals dict.

        If *state* is None and *market_data* is provided, builds one via
        ``UnifiedState.from_runtime()``.

        Returns a dict with ``model_alpha_*`` keys, or a neutral dict with
        ``model_alpha_data_quality=0`` if gated/failed.
        """
        if obs_126 is None and market_data is not None and ohlcv_df is not None:
            _obs_builder = obs_builder or self._get_obs_builder()
            if _obs_builder is not None:
                try:
                    obs_126 = _obs_builder.build_obs(
                        asset=asset,
                        ohlcv_df=ohlcv_df,
                        market_data=market_data,
                        gmm_probs=gmm_probs if gmm_probs is not None else market_data.get("_gmm_probs", []),
                        env_state=env_state or {},
                    )
                except Exception as e:
                    logger.debug(f"[MODEL_ALPHA] build_obs skipped for {asset}: {e}")

        if state is None and market_data is not None:
            state = UnifiedState.from_runtime(
                primary_asset=asset,
                market_data=market_data,
                agent_signals=agent_signals,
                positions=positions,
                obs_126=obs_126,
            )
        elif state is not None and obs_126 is not None:
            try:
                state._obs_126_by_asset[asset] = np.asarray(obs_126, dtype=np.float32)
            except Exception:
                pass

        if state is None:
            return self._neutral_agent_signals(asset, "no_state")

        intent = self.generate_intent(state, asset)
        if intent is None:
            reason = getattr(self, "_last_gating_reason", "filtered")
            backend = self._model_backend_by_asset.get(asset, {})
            return self._neutral_agent_signals(
                asset,
                reason,
                model_type=str(backend.get("model_type", "none") or "none"),
                model_version=str(backend.get("model_version", "none") or "none"),
                data_quality=max(0.0, min(1.0, float(getattr(state, "_coverage", 1.0) or 0.0))),
                data_age_seconds=float(getattr(state, "_data_age_seconds", 0.0) or 0.0),
            )

        sig = intent.to_agent_signals()
        sig["model_alpha_weight"] = round(
            self._current_weight * float(getattr(intent, "effective_weight", 1.0) or 1.0),
            4,
        )
        sig["model_alpha_data_quality"] = max(0.0, min(1.0, getattr(state, "_coverage", 1.0)))
        sig["model_alpha_authority"] = "CONTEXT"
        return sig

    # v6 contract aliases
    def generate_signal(
        self,
        asset: str = "BTC",
        market_data: Optional[Dict] = None,
        agent_signals: Optional[Dict] = None,
        positions: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        V6 agent contract - alias for to_agent_signals().

        Accepts the same kwargs main.py passes to every agent.
        """
        return self.to_agent_signals(
            asset=asset,
            market_data=market_data,
            agent_signals=agent_signals,
            positions=positions,
        )

    def to_agent_signal_dict(
        self,
        asset: str = "BTC",
        market_data: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Alias for generate_signal() - v6 naming convention."""
        return self.generate_signal(asset=asset, market_data=market_data, **kwargs)

    @staticmethod
    def _neutral_agent_signals(
        asset: str,
        reason: str,
        model_type: str = "none",
        model_version: str = "none",
        data_quality: float = 0.0,
        data_age_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat() + "Z"
        # Map gating reason to data_quality string
        dq_map = {
            "no_state": "missing",
            "model_not_loaded": "missing",
            "unknown_asset": "error",
        }
        dq = dq_map.get(reason, "error" if "error" in reason else "missing")
        return {
            "model_alpha_direction": 0.0,
            "model_alpha_confidence": 0.0,
            "model_alpha_raw_prediction": 0.0,
            "model_alpha_horizon": "none",
            "model_alpha_weight": 0.0,
            "model_alpha_asof_timestamp": now_iso,
            "model_alpha_data_quality": max(0.0, min(1.0, float(data_quality or 0.0))),
            "model_alpha_reasons": [f"gated:{reason}"],
            "model_alpha_model_type": str(model_type or "none"),
            "model_alpha_model_version": str(model_version or "none"),
            # v6.4 standard metadata
            "asof_ts": now_iso,
            "data_age_seconds": float(data_age_seconds or 0.0),
            "data_quality": dq,
            "provenance": ["model_alpha_agent", asset, f"gated:{reason}"],
        }


# =============================================================================
# REAL DECISION TRANSFORMER WRAPPER
# =============================================================================

class RealDecisionTransformer:
    """
    Decision Transformer v3.2 Runtime Wrapper

    Loads a trained DecisionTransformerV32 model (GPT2-based) and provides
    inference compatible with the ModelAlphaAgent / DRL Ensemble interface.

    State dim = 126 (122 features + 4 env state), aligned with TQC obs_dim.
    """

    def __init__(
        self,
        model_path: str,
        state_dim: int = 126,
        action_dim: int = 1,
        hidden_dim: int = 384,
    ):
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._context_buffer = []  # Rolling buffer for sequence input
        self._context_length = 96  # Default, overridden from checkpoint config
        self._scaler = None  # RobustScaler for feature scaling
        self._scale_col_indices = None  # Indices of scalable features (not regime_proba/has_external)

        self._load_model()
        self._load_scaler()

    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float32)

    def _load_model(self):
        """Load the DecisionTransformerV32 model from checkpoint."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        try:
            from training.drl.train_decision_transformer_v32 import (
                DTConfigV32, DecisionTransformerV32,
            )
            import __main__ as _runtime_main

            # Older checkpoints pickle DTConfigV32 under __main__.DTConfigV32.
            if not hasattr(_runtime_main, "DTConfigV32"):
                setattr(_runtime_main, "DTConfigV32", DTConfigV32)

            checkpoint = torch.load(self.model_path, map_location=self.device,
                                    weights_only=False)

            # Reconstruct config from checkpoint
            saved_config = checkpoint.get('config')
            if saved_config is not None:
                if hasattr(saved_config, 'state_dim'):
                    # saved_config is a DTConfigV32 dataclass
                    config = saved_config
                else:
                    # saved_config is a dict
                    config = DTConfigV32(**{k: v for k, v in saved_config.items()
                                           if k in DTConfigV32.__dataclass_fields__})
                self.state_dim = config.state_dim
                self._context_length = config.context_length
                self.model = DecisionTransformerV32(config)
            else:
                # Fallback: build from saved state_dim
                config = DTConfigV32(
                    state_dim=self.state_dim,
                    hidden_size=self.hidden_dim,
                )
                self.model = DecisionTransformerV32(config)

            # Load weights
            state_key = 'model_state' if 'model_state' in checkpoint else 'model_state_dict'
            if state_key in checkpoint:
                self.model.load_state_dict(checkpoint[state_key], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

            self.model.eval()
            self.model.to(self.device)
            logger.info(f"DT v3.2 loaded: state_dim={self.state_dim}, "
                        f"ctx={self._context_length}, device={self.device}")

        except Exception as e:
            logger.error(f"Failed to load Decision Transformer: {e}")
            self.model = None

    def _load_scaler(self):
        """Load the DT feature scaler from the same directory as the model.

        Scaler format: raw RobustScaler saved via joblib.dump().
        Scale indices: first 113 features are scalable, last 9 (regime_proba_0..7
        + has_external_data) are passed through.
        """
        import os
        model_dir = os.path.dirname(self.model_path)
        scaler_path = os.path.join(model_dir, "dt_scaler.pkl")
        if not os.path.exists(scaler_path):
            return

        try:
            import joblib
            scaler_obj = joblib.load(scaler_path)
            if isinstance(scaler_obj, dict):
                self._scaler = scaler_obj.get("scaler")
            else:
                self._scaler = scaler_obj

            # Determine which feature indices to scale (skip last 9: regime_proba + has_external)
            n_features = 122  # 122 features in obs, remaining 4 are env state
            n_no_scale = 9    # regime_proba_0..7 + has_external_data
            self._scale_col_indices = list(range(n_features - n_no_scale))  # [0..112]
            logger.info(f"DT scaler loaded: {scaler_path} ({len(self._scale_col_indices)} scalable features)")
        except Exception as e:
            logger.warning(f"Failed to load DT scaler: {e}")

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Make single-step prediction using the DT model.

        Maintains a rolling context buffer. The model predicts the next action
        given the sequence of past states.

        Args:
            features: 126-dim observation vector (same as TQC obs).

        Returns:
            (prediction, confidence): prediction in [-1, 1], confidence in [0, 1]
        """
        if self.model is None:
            # [FIX-AG7] Was (0.0, 0.0) — indistinguishable from genuine neutral.
            # Return None to signal "inference unavailable" to caller.
            return None, None

        try:
            # Apply scaler if available (scale first 113 features, clip to [-10,10])
            features = features.copy()
            if self._scaler is not None and self._scale_col_indices is not None:
                scalable = features[self._scale_col_indices].reshape(1, -1)
                features[self._scale_col_indices] = np.clip(
                    self._scaler.transform(scalable).flatten(), -10.0, 10.0
                )

            # Add to context buffer
            self._context_buffer.append(features)
            if len(self._context_buffer) > self._context_length:
                self._context_buffer = self._context_buffer[-self._context_length:]

            ctx_len = len(self._context_buffer)

            with torch.no_grad():
                # Build sequence tensors
                states = torch.tensor(
                    np.array(self._context_buffer), dtype=torch.float32,
                    device=self.device
                ).unsqueeze(0)  # (1, ctx, 126)

                # Dummy actions and RTG (zeros - we only need the state branch)
                actions = torch.zeros(1, ctx_len, self.action_dim,
                                      device=self.device)
                # Simple RTG: small positive target
                rtg = torch.ones(1, ctx_len, 4, device=self.device) * 0.01
                timesteps = torch.arange(ctx_len, device=self.device).unsqueeze(0)

                output = self.model(states, actions, rtg, timesteps)

                # Take last timestep prediction
                action_pred = output['action_preds'][0, -1, 0]
                prediction = float(action_pred.cpu().numpy())
                prediction = max(-1.0, min(1.0, prediction))

                # Direction logits -> confidence
                dir_logits = output['direction_logits'][0, -1]  # (3,)
                dir_probs = torch.softmax(dir_logits, dim=0)
                confidence = float(dir_probs.max().cpu().numpy())

                return prediction, confidence

        except Exception as e:
            logger.error(f"DT inference failed: {e}")
            return None, None  # [FIX-AG7] Signal failure, not neutral


class RealSequenceAlphaModel:
    """Supervised sequence alpha model wrapper.

    Consumes the first 122 scaled runtime features from the 126-dim obs and
    outputs a directional advisory signal with confidence.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.model = None
        self.state_dim = 122
        self._context_length = 48
        self._target_scale_bps = 64.0
        self._context_buffer = []
        self._load_model()

    def _load_model(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        from training.model_alpha.sequence_alpha_model import SequenceAlphaConfig, SequenceAlphaNet

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        saved_config = checkpoint.get("config", {})
        if isinstance(saved_config, SequenceAlphaConfig):
            config = saved_config
        else:
            config = SequenceAlphaConfig(**saved_config)

        self.state_dim = int(config.input_dim)
        self._context_length = int(config.seq_len)
        self._target_scale_bps = float(config.target_scale_bps)
        self.model = SequenceAlphaNet(config)
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        self.model.eval()
        self.model.to(self.device)
        logger.info(
            f"Sequence alpha loaded: state_dim={self.state_dim}, seq_len={self._context_length}, device={self.device}"
        )

    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if len(features) >= self.state_dim:
            return features[: self.state_dim]
        return features

    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        if self.model is None:
            return None, None  # [FIX-AG7]

        try:
            features = self.prepare_features(features)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            self._context_buffer.append(features)
            if len(self._context_buffer) > self._context_length:
                self._context_buffer = self._context_buffer[-self._context_length:]

            if len(self._context_buffer) < self._context_length:
                pad = [self._context_buffer[0]] * (self._context_length - len(self._context_buffer))
                seq = np.asarray(pad + self._context_buffer, dtype=np.float32)
            else:
                seq = np.asarray(self._context_buffer, dtype=np.float32)

            with torch.no_grad():
                x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                out = self.model(x)
                probs = torch.softmax(out["logits"][0], dim=0)
                cls = int(torch.argmax(probs).item())  # 0=short,1=flat,2=long
                confidence = float(torch.max(probs).item())
                signed_edge = float(torch.tanh(out["value"][0]).item())

            if cls == 2:
                prediction = abs(signed_edge)
            elif cls == 0:
                prediction = -abs(signed_edge)
            else:
                prediction = 0.0

            prediction = max(-1.0, min(1.0, prediction))
            confidence = max(0.0, min(1.0, confidence))
            return prediction, confidence
        except Exception as e:
            logger.error(f"Sequence alpha inference failed: {e}")
            return None, None  # [FIX-AG7]


# =============================================================================
# MOCK MODEL (用于开发/测试)
# =============================================================================

class MockDecisionTransformer:
    """
    Mock Decision Transformer
    
    生产环境需替换为真实模型
    """
    
    def __init__(self):
        self._call_count = 0
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Mock 预测
        
        Returns:
            (prediction, confidence): prediction 范围 [-1, 1]
        """
        self._call_count += 1
        
        # 基于特征的简单规则（模拟模型行为）
        # features: [ret_1h, ret_24h, vol, lob_imb, spread, whale, exch_flow, fg, sent, fund, pos, risk, usd]
        
        signal = 0.0
        
        # 动量信号
        ret_1h = features[0]
        ret_24h = features[1]
        signal += ret_1h * 0.3 + ret_24h * 0.2
        
        # 订单簿失衡
        lob_imb = features[3]
        signal += lob_imb * 0.2
        
        # 情绪
        fg = features[7]  # fear_greed normalized
        sent = features[8]
        signal += (fg + sent) * 0.15
        
        # 逆向策略：极端恐惧买入，极端贪婪卖出
        if fg < -0.6:  # 极端恐惧
            signal += 0.2
        elif fg > 0.6:  # 极端贪婪
            signal -= 0.2
        
        # 限制范围
        signal = max(-1.0, min(1.0, signal))
        
        # 置信度基于信号强度
        confidence = min(0.9, 0.5 + abs(signal) * 0.5)
        
        return signal, confidence


# =============================================================================
# SINGLETON
# =============================================================================

_model_alpha_agent_instance: Optional[ModelAlphaAgent] = None


def get_model_alpha_agent(
    model_type: ModelType = ModelType.AUTO,
    assets: List[str] = None,
    allow_mock_model: bool = False,
) -> ModelAlphaAgent:
    """Get ModelAlphaAgent singleton."""
    global _model_alpha_agent_instance
    normalized_assets = list(assets or ["BTC", "ETH", "SOL"])
    if _model_alpha_agent_instance is None:
        _model_alpha_agent_instance = ModelAlphaAgent(
            model_type=model_type,
            assets=normalized_assets,
            allow_mock_model=allow_mock_model,
        )
    elif (
        _model_alpha_agent_instance.model_type != model_type
        or list(_model_alpha_agent_instance.assets) != normalized_assets
        or bool(_model_alpha_agent_instance.allow_mock_model) != bool(allow_mock_model)
    ):
        logger.info("ModelAlphaAgent constructor inputs changed; refreshing singleton instance")
        _model_alpha_agent_instance = ModelAlphaAgent(
            model_type=model_type,
            assets=normalized_assets,
            allow_mock_model=allow_mock_model,
        )
    return _model_alpha_agent_instance


def reset_model_alpha_agent():
    """重置单例"""
    global _model_alpha_agent_instance
    _model_alpha_agent_instance = None

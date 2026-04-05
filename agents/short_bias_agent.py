"""
================================================================================
HMATS v6.2.3 - SHORT-BIASED AGENT (做空专用智能体)
================================================================================
Version: 2.0.0
Purpose: 专门识别并进场做空机会的策略 Agent

[WARN]️ 关键约束：
    - 该 Agent 只输出做空建议，永远不做多
    - direction 固定为 -1 或 0，永远不为正
    - 不下单、不执行，只输出信号
    - 必须通过系统决策融合层 (Authority Fusion)
    - 遵守一票否决逻辑 (One-Veto-Kill)

做空机会识别：
    1. 极端上涨后的反转结构 (Parabolic Top)
    2. 资金费率极度正 (Extreme Positive Funding - 多头拥挤)
    3. 清算爆仓堆积前高杠杆持仓 (Pre-Liquidation Cascade)
    4. 高频社交媒体贪婪言论 (Extreme Greed Sentiment)
    5. 链上大额流出 (Whale Outflow)

接口兼容：
    - 输出与 AgentSignal 兼容
    - 可被 AuthorityFusionEngine 消费
    - 遵守 Authority Matrix 权限

v2.0.0 changes:
    - Per-asset state isolation (P0)
    - Timestamped history with time-based lookbacks (P0)
    - Ingest-before-gating (P0)
    - Short-only enforcement: direction > 0 path removed (P0)
    - Input normalization (fear_greed, funding_rate) (P1)
    - Structured diagnostics in to_agent_signal_dict (P1)
    - Config window_hours fields actually used (P1)
    - Optional regime gating (P2)

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ShortBiasConfig:
    """Short-biased Agent 配置"""

    # === 输出限制 ===
    max_confidence: float = 0.85
    min_confidence_threshold: float = 0.40
    min_observation_minutes: int = 60

    # === Parabolic Top 参数 ===
    parabolic_price_surge_threshold: float = 0.08  # 8% 短期涨幅
    parabolic_price_surge_window_hours: int = 24
    parabolic_volume_spike_ratio: float = 2.5
    parabolic_rsi_extreme: float = 80.0

    # === Funding Rate 参数 ===
    funding_high_threshold: float = 0.0005   # 0.05% = 高
    funding_extreme_threshold: float = 0.001  # 0.1% = 极端
    funding_trend_window_hours: int = 24

    # === Funding Rate Arbitrage (v6.7) ===
    funding_arb_short_boost_threshold: float = 0.0003  # [FIX-AG13] Was 0.0024 (never fired). Kraken 8h rate range 0.0001-0.0008. 0.0003 = meaningful carry.
    funding_arb_short_confidence_boost: float = 0.15    # +15% confidence for shorts

    # === 清算堆积参数 ===
    liquidation_long_spike_ratio: float = 3.0
    liquidation_oi_growth_threshold: float = 0.10
    liquidation_lookback_hours: float = 24.0

    # === 情绪参数 ===
    greed_extreme_threshold: float = 0.80
    fear_greed_lookback_hours: int = 48

    # === 链上参数 ===
    whale_outflow_threshold_usd: float = 50_000_000
    whale_alert_window_hours: int = 12

    # === [UPG2] Structure Break 参数 ===
    structure_break_lookback_hours: float = 168.0  # 7 天
    structure_break_threshold: float = 0.01  # 1% 跌破视为 break
    structure_break_min_bars: int = 42  # 至少 7 天 × 6 bar/天

    # === 信号权重 ===
    weight_parabolic: float = 0.25
    weight_funding: float = 0.20
    weight_liquidation: float = 0.15
    weight_sentiment: float = 0.15
    weight_whale: float = 0.10
    weight_structure_break: float = 0.15  # [UPG2]

    # === 风险限制 ===
    signal_cooldown_minutes: int = 30
    max_consecutive_signals: int = 3

    # === Volume average window (hours) ===
    volume_avg_window_hours: float = 400.0

    # === Regime gating (P2) ===
    regime_quiet_penalty: float = 0.5   # confidence multiplier in quiet regimes
    regime_quiet_names: Tuple[str, ...] = ("QUIET", "LOW_VOL", "CONSOLIDATION")


# =============================================================================
# SIGNAL COMPONENTS
# =============================================================================

class ShortSignalType(Enum):
    """做空信号类型"""
    NONE = "NONE"
    PARABOLIC_TOP = "PARABOLIC_TOP"
    EXTREME_FUNDING = "EXTREME_FUNDING"
    LIQUIDATION_CASCADE = "LIQUIDATION_CASCADE"
    EXTREME_GREED = "EXTREME_GREED"
    WHALE_OUTFLOW = "WHALE_OUTFLOW"
    STRUCTURE_BREAK = "STRUCTURE_BREAK"  # [UPG2] 支撑位跌破
    COMPOSITE = "COMPOSITE"


@dataclass
class ShortSignalComponent:
    """单个做空信号组件"""
    signal_type: ShortSignalType
    raw_score: float  # 0 to 1
    confidence: float  # 0 to 1
    reason: str
    triggered: bool
    data: Dict = field(default_factory=dict)


@dataclass
class ShortBiasSignal:
    """
    Short-biased Agent 输出信号

    与 AgentSignal 接口兼容，可被 Authority Fusion 消费
    """
    # === 核心输出 ===
    direction: float = 0.0   # -1 (short) or 0 (neutral). NEVER positive.
    confidence: float = 0.0  # 0 to max_confidence

    # === 信号元数据 ===
    primary_trigger: ShortSignalType = ShortSignalType.NONE
    components: List[ShortSignalComponent] = field(default_factory=list)

    # === 风险提示 ===
    veto_active: bool = False
    veto_direction: Optional[str] = None  # DEPRECATED: short-only agent, always None

    # === 时间戳 ===
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # === 建议 ===
    suggested_stop_pct: float = 0.03
    max_exposure_pct: float = 0.30

    # === 调试 ===
    reasons: List[str] = field(default_factory=list)

    # === v2.0: Asset + diagnostics ===
    asset: str = "BTC"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_agent_signal_dict(self) -> Dict:
        """转换为与 AgentSignal 兼容的格式"""
        return {
            # Existing keys (backward compatible)
            "direction": self.direction,
            "confidence": self.confidence,
            "veto_active": self.veto_active,
            "veto_direction": self.veto_direction,
            "leverage_cap": 1.0,
            "exit_pressure": 0.0,
            "tranche_advice": None,
            # v2.0 diagnostic keys (additive, safe for existing consumers)
            "asset": self.asset,
            "asof_ts": self.timestamp.isoformat(),
            "meta": self.meta,
        }

    def is_actionable(self) -> bool:
        """信号是否可操作 - short-only: direction must be -1"""
        return self.direction < 0 and self.confidence > 0


# =============================================================================
# SHORT-BIASED AGENT
# =============================================================================

class ShortBiasAgent:
    """
    做空专用智能体

    核心原则：
    1. 只做空，永远不做多 (direction only -1 or 0)
    2. 只输出信号，不执行
    3. 遵守一票否决
    4. 保守置信度上限
    5. Per-asset state isolation
    6. Always ingest history before gating

    使用方式：
        agent = ShortBiasAgent()
        signal = agent.analyze(market_data, asset="BTC")
    """

    def __init__(self, config: ShortBiasConfig = None):
        self.config = config or ShortBiasConfig()

        # Per-asset timestamped histories: deque of (datetime, value)
        self._price_history: Dict[str, deque] = {}     # (ts, float)
        self._volume_history: Dict[str, deque] = {}    # (ts, float)
        self._funding_history: Dict[str, deque] = {}   # (ts, dict)
        self._liquidation_history: Dict[str, deque] = {}  # (ts, dict)
        self._sentiment_history: Dict[str, deque] = {}  # (ts, dict)

        # Per-asset gating state
        self._last_signal_time: Dict[str, Optional[datetime]] = {}
        self._consecutive_signals: Dict[str, int] = {}
        self._observation_start: Dict[str, datetime] = {}

        # Per-asset stats
        self._total_signals: Dict[str, int] = {}
        self._signal_breakdown: Dict[str, Dict[ShortSignalType, int]] = {}

        logger.info("[SHORT_BIAS] Agent initialized - ONLY SHORT SIGNALS (v2.0 per-asset)")

    def _ensure_asset(self, asset: str, now: datetime):
        """Lazily initialize per-asset state."""
        if asset not in self._price_history:
            self._price_history[asset] = deque(maxlen=2000)
            self._volume_history[asset] = deque(maxlen=2000)
            self._funding_history[asset] = deque(maxlen=500)
            self._liquidation_history[asset] = deque(maxlen=500)
            self._sentiment_history[asset] = deque(maxlen=500)
            self._last_signal_time[asset] = None
            self._consecutive_signals[asset] = 0
            self._observation_start[asset] = now
            self._total_signals[asset] = 0
            self._signal_breakdown[asset] = {t: 0 for t in ShortSignalType}

    # =========================================================================
    # MARKET DATA KEY ALIASES
    # =========================================================================

    @staticmethod
    def _resolve_market_keys(market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve common key aliases in market_data.

        Callers may pass 'close' instead of 'price', 'vol_24h' instead of
        'volume', etc.  This normalizes without mutating the original dict.
        """
        resolved = dict(market_data)
        # price aliases
        if "price" not in resolved:
            for alias in ("close", "last_price", "mark_price"):
                if alias in resolved:
                    resolved["price"] = resolved[alias]
                    break
        # volume aliases
        if "volume" not in resolved:
            for alias in ("vol_24h", "volume_24h", "base_volume"):
                if alias in resolved:
                    resolved["volume"] = resolved[alias]
                    break
        # rsi aliases
        if "rsi" not in resolved:
            for alias in ("rsi_14", "rsi14"):
                if alias in resolved:
                    resolved["rsi"] = resolved[alias]
                    break
        return resolved

    # =========================================================================
    # HMATS COMPATIBILITY - generate_signal
    # =========================================================================

    def generate_signal(
        self,
        asset: str = "BTC",
        price: float = 0.0,
        regime: str = "",
        market_data: Optional[Dict] = None,
        derivatives_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        onchain_data: Optional[Dict] = None,
    ) -> "ShortBiasSignal":
        """
        Unified entry point matching HMATS generate_signal() convention.

        Always returns ShortBiasSignal (never None, never raises).
        """
        mkt = market_data or {"price": price}
        return self.analyze(
            market_data=mkt,
            derivatives_data=derivatives_data,
            sentiment_data=sentiment_data,
            onchain_data=onchain_data,
            asset=asset,
            regime_state=regime or None,
        )

    # =========================================================================
    # TIME-BASED LOOKBACK HELPERS
    # =========================================================================

    @staticmethod
    def _get_lookback_series(history: deque, now: datetime, lookback_hours: float) -> List[Tuple[datetime, Any]]:
        """Get (ts, value) entries within [now - lookback_hours, now]."""
        cutoff = now - timedelta(hours=lookback_hours)
        return [(ts, v) for ts, v in history if ts >= cutoff]

    @staticmethod
    def _get_value_at_or_before(history: deque, target_time: datetime) -> Any:
        """Get value from the entry closest to (but not after) target_time."""
        best_val = None
        for ts, v in history:
            if ts <= target_time:
                best_val = v
            elif ts > target_time:
                break  # deque is chronologically ordered
        return best_val

    # =========================================================================
    # INPUT NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_fear_greed(raw: float) -> Tuple[float, str]:
        """Normalize fear/greed to [0, 1]. Auto-detects 0-100 vs 0-1 scale."""
        if raw > 1.5:
            return max(0.0, min(raw / 100.0, 1.0)), "assumed_0_100_scale"
        return max(0.0, min(raw, 1.0)), "assumed_0_1_scale"

    @staticmethod
    def _normalize_funding_rate(raw: float) -> Tuple[float, str]:
        """Record funding rate interpretation. No transform - just record unit assumption."""
        # Convention: funding_rate is per-8h (Kraken/Binance standard)
        # Values typically 0.0001 = 0.01%/8h
        return raw, "per_8h"

    # =========================================================================
    # CENTRALIZED INGESTION (P0: always called before gating)
    # =========================================================================

    def _ingest(
        self,
        asset: str,
        now: datetime,
        market_data: Dict[str, Any],
        derivatives_data: Optional[Dict],
        sentiment_data: Optional[Dict],
        onchain_data: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Ingest all data into per-asset timestamped histories.
        Returns meta dict with normalization info and missing inputs.
        """
        meta: Dict[str, Any] = {
            "asset": asset,
            "asof_ts": now.isoformat(),
            "missing_inputs": [],
            "normalization": {},
            "data_age_seconds": {},
        }

        # --- Price ---
        price = market_data.get("price", 0)
        if price > 0:
            self._price_history[asset].append((now, price))

        # --- Volume ---
        volume = market_data.get("volume", 0)
        if volume > 0:
            self._volume_history[asset].append((now, volume))

        # --- Derivatives (funding, OI, liquidations) ---
        if derivatives_data:
            funding_raw = derivatives_data.get("funding_rate", 0)
            funding_norm, funding_unit = self._normalize_funding_rate(funding_raw)
            meta["normalization"]["funding_rate"] = {
                "raw": funding_raw, "normalized": funding_norm, "unit": funding_unit,
            }
            open_interest = derivatives_data.get("open_interest", 0)
            self._funding_history[asset].append((now, {
                "funding_rate": funding_norm,
                "open_interest": open_interest,
            }))

            liq_long = derivatives_data.get("liquidations_long", 0)
            liq_short = derivatives_data.get("liquidations_short", 0)
            self._liquidation_history[asset].append((now, {
                "long": liq_long, "short": liq_short, "oi": open_interest,
            }))
        else:
            meta["missing_inputs"].append("derivatives_data")

        # --- Sentiment ---
        if sentiment_data:
            fg_raw = sentiment_data.get("fear_greed_index", 0.5)
            fg_norm, fg_interp = self._normalize_fear_greed(fg_raw)
            meta["normalization"]["fear_greed"] = {
                "raw": fg_raw, "normalized": fg_norm, "interpretation": fg_interp,
            }
            social = sentiment_data.get("social_sentiment", 0.0)
            self._sentiment_history[asset].append((now, {
                "fg": fg_norm, "social": social,
            }))
        else:
            meta["missing_inputs"].append("sentiment_data")

        # --- Onchain ---
        if not onchain_data:
            meta["missing_inputs"].append("onchain_data")

        # --- Data age for most recent entries ---
        for name, hist in [
            ("price", self._price_history[asset]),
            ("funding", self._funding_history[asset]),
            ("sentiment", self._sentiment_history[asset]),
        ]:
            if hist:
                meta["data_age_seconds"][name] = round(
                    (now - hist[-1][0]).total_seconds(), 1
                )

        return meta

    # =========================================================================
    # GATING (applied to output only, AFTER ingestion)
    # =========================================================================

    def _check_gates(self, asset: str, now: datetime) -> Optional[str]:
        """Check observation/cooldown/consecutive gates. Returns gate reason or None."""
        # Observation period
        obs_minutes = (now - self._observation_start[asset]).total_seconds() / 60
        if obs_minutes < self.config.min_observation_minutes:
            return f"observation:{obs_minutes:.0f}/{self.config.min_observation_minutes}min"

        # Cooldown
        last = self._last_signal_time.get(asset)
        if last:
            cooldown_elapsed = (now - last).total_seconds() / 60
            if cooldown_elapsed < self.config.signal_cooldown_minutes:
                return f"cooldown:{cooldown_elapsed:.0f}/{self.config.signal_cooldown_minutes}min"

        # Max consecutive
        consec = self._consecutive_signals.get(asset, 0)
        if consec >= self.config.max_consecutive_signals:
            self._consecutive_signals[asset] = 0
            return f"max_consecutive:{consec}/{self.config.max_consecutive_signals}"

        return None

    def _gated_signal(
        self, asset: str, now: datetime, gate_reason: str, meta: Dict,
    ) -> ShortBiasSignal:
        """Return a neutral signal with gate diagnostics."""
        meta["gates"] = [gate_reason]
        return ShortBiasSignal(
            timestamp=now,
            asset=asset,
            reasons=[f"Gated: {gate_reason}"],
            meta=meta,
        )

    # =========================================================================
    # CORE ANALYSIS
    # =========================================================================

    def analyze(
        self,
        market_data: Dict[str, Any],
        derivatives_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        onchain_data: Optional[Dict] = None,
        *,
        asset: Optional[str] = None,
        regime_state: Optional[str] = None,
        regime_confidence: float = 0.0,
    ) -> ShortBiasSignal:
        """
        分析市场数据，输出做空信号

        Args:
            market_data: 价格/成交量数据 (price, volume, high_24h, low_24h, rsi)
            derivatives_data: 衍生品数据 (funding_rate, open_interest, liquidations_*)
            sentiment_data: 情绪数据 (fear_greed_index, social_sentiment)
            onchain_data: 链上数据 (whale_flow_usd, exchange_inflow)
            asset: Asset identifier (inferred from market_data["asset"] if absent)
            regime_state: Optional regime name for P2 gating
            regime_confidence: Regime classifier confidence

        Returns:
            ShortBiasSignal: direction only -1 or 0, never positive
        """
        try:
            return self._analyze_inner(
                market_data, derivatives_data, sentiment_data, onchain_data,
                asset=asset, regime_state=regime_state, regime_confidence=regime_confidence,
            )
        except Exception as e:
            logger.error(f"[SHORT_BIAS] analyze() exception: {e}", exc_info=True)
            return ShortBiasSignal(
                timestamp=datetime.now(timezone.utc),
                asset=asset or market_data.get("asset", "BTC"),
                reasons=[f"internal_error: {e}"],
                meta={"error": str(e)},
            )

    def _analyze_inner(
        self,
        market_data: Dict[str, Any],
        derivatives_data: Optional[Dict],
        sentiment_data: Optional[Dict],
        onchain_data: Optional[Dict],
        *,
        asset: Optional[str],
        regime_state: Optional[str],
        regime_confidence: float,
    ) -> ShortBiasSignal:
        asset = asset or market_data.get("asset", "BTC")
        now = datetime.now(timezone.utc)
        self._ensure_asset(asset, now)

        # Resolve key aliases before processing
        market_data = self._resolve_market_keys(market_data)

        # 1. ALWAYS ingest first (P0)
        meta = self._ingest(asset, now, market_data, derivatives_data, sentiment_data, onchain_data)
        meta["gates"] = []

        # 2. Apply gating (output only - history already updated)
        gate = self._check_gates(asset, now)
        if gate:
            return self._gated_signal(asset, now, gate, meta)

        # 3. Analyze components (per-asset, time-based)
        components: List[ShortSignalComponent] = [
            self._analyze_parabolic_top(asset, now, market_data),
            self._analyze_extreme_funding(asset, now, derivatives_data),
            self._analyze_liquidation_cascade(asset, now, derivatives_data),
            self._analyze_extreme_greed(asset, now, sentiment_data),
            self._analyze_whale_outflow(onchain_data),
            self._analyze_structure_break(asset, now, market_data),  # [UPG2]
        ]

        # 4. Synthesize (short-only)
        signal = self._synthesize_signal(asset, now, components, meta)

        # 5. Funding arb boost (short-only: only boosts shorts, never creates longs)
        self._apply_funding_arb(signal, derivatives_data)

        # 6. Regime gating (P2, optional)
        if regime_state is not None and regime_confidence > 0.5:
            self._apply_regime_gating(signal, regime_state, regime_confidence)

        # 7. Update gating state
        if signal.is_actionable():
            self._last_signal_time[asset] = now
            self._consecutive_signals[asset] = self._consecutive_signals.get(asset, 0) + 1
            self._total_signals[asset] = self._total_signals.get(asset, 0) + 1
            self._signal_breakdown[asset][signal.primary_trigger] += 1
            logger.info(
                f"[SHORT_BIAS] {asset}: direction={signal.direction}, "
                f"confidence={signal.confidence:.2f}, "
                f"trigger={signal.primary_trigger.value}"
            )
        else:
            self._consecutive_signals[asset] = 0

        return signal

    # =========================================================================
    # COMPONENT ANALYZERS (pure readers - no history mutation)
    # =========================================================================

    def _analyze_parabolic_top(self, asset: str, now: datetime, market_data: Dict) -> ShortSignalComponent:
        """检测抛物线顶部形态"""
        component = ShortSignalComponent(
            signal_type=ShortSignalType.PARABOLIC_TOP,
            raw_score=0.0, confidence=0.0, reason="", triggered=False,
        )
        try:
            price = market_data.get("price", 0)
            high_24h = market_data.get("high_24h", price)
            volume = market_data.get("volume", 0)
            rsi = market_data.get("rsi", 50)

            # Price change using time-based lookback (P0: uses config window_hours)
            window_hours = self.config.parabolic_price_surge_window_hours
            price_then = self._get_value_at_or_before(
                self._price_history[asset],
                now - timedelta(hours=window_hours),
            )
            price_change = (price - price_then) / price_then if price_then and price_then > 0 else 0

            # Volume ratio using time-based lookback
            vol_entries = self._get_lookback_series(
                self._volume_history[asset], now, self.config.volume_avg_window_hours,
            )
            if len(vol_entries) >= 5:
                avg_volume = np.mean([v for _, v in vol_entries])
                volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            else:
                volume_ratio = 1

            scores = []
            reasons = []

            # 1. Price surge
            if price_change > self.config.parabolic_price_surge_threshold:
                surge_score = min(1.0, price_change / (self.config.parabolic_price_surge_threshold * 2))
                scores.append(surge_score)
                reasons.append(f"Price surge: +{price_change:.1%}")

            # 2. Volume spike
            if volume_ratio > self.config.parabolic_volume_spike_ratio:
                vol_score = min(1.0, (volume_ratio - 1) / (self.config.parabolic_volume_spike_ratio - 1))
                scores.append(vol_score)
                reasons.append(f"Volume spike: {volume_ratio:.1f}x")

            # 3. RSI overbought
            if rsi > self.config.parabolic_rsi_extreme:
                rsi_score = min(1.0, (rsi - self.config.parabolic_rsi_extreme) / 20)
                scores.append(rsi_score)
                reasons.append(f"RSI overbought: {rsi:.1f}")

            # 4. Price retreat from high
            if high_24h > 0 and price < high_24h * 0.98:
                retreat_pct = (high_24h - price) / high_24h
                retreat_score = min(1.0, retreat_pct / 0.05)
                scores.append(retreat_score)
                reasons.append(f"Price retreat from high: -{retreat_pct:.1%}")

            if len(scores) >= 2:
                component.raw_score = np.mean(scores)
                component.confidence = min(component.raw_score, 0.8)
                component.triggered = component.confidence > 0.4
                component.reason = "; ".join(reasons)
                component.data = {"price_change": price_change, "volume_ratio": volume_ratio, "rsi": rsi}

        except Exception as e:
            logger.debug(f"Parabolic analysis error: {e}")

        return component

    def _analyze_extreme_funding(self, asset: str, now: datetime, derivatives_data: Optional[Dict]) -> ShortSignalComponent:
        """检测极端正资金费率"""
        component = ShortSignalComponent(
            signal_type=ShortSignalType.EXTREME_FUNDING,
            raw_score=0.0, confidence=0.0, reason="", triggered=False,
        )
        if not derivatives_data:
            return component

        try:
            funding_rate = derivatives_data.get("funding_rate", 0)
            open_interest = derivatives_data.get("open_interest", 0)

            scores = []
            reasons = []

            # 1. Current funding level
            if funding_rate > self.config.funding_extreme_threshold:
                level_score = min(1.0, funding_rate / (self.config.funding_extreme_threshold * 2))
                scores.append(level_score * 1.5)
                reasons.append(f"Extreme funding: {funding_rate:.4%}")
            elif funding_rate > self.config.funding_high_threshold:
                level_score = (funding_rate - self.config.funding_high_threshold) / (
                    self.config.funding_extreme_threshold - self.config.funding_high_threshold
                )
                scores.append(level_score)
                reasons.append(f"High funding: {funding_rate:.4%}")

            # 2. Funding trend (time-based, P0)
            funding_window = self._get_lookback_series(
                self._funding_history[asset], now, self.config.funding_trend_window_hours,
            )
            if len(funding_window) >= 2:
                trend = funding_window[-1][1]["funding_rate"] - funding_window[0][1]["funding_rate"]
                if trend > 0.0001:
                    trend_score = min(1.0, trend / 0.0003)
                    scores.append(trend_score)
                    reasons.append(f"Funding trending up: +{trend:.4%}")

            # 3. OI change (time-based, P0)
            oi_then = self._get_value_at_or_before(
                self._funding_history[asset], now - timedelta(hours=24),
            )
            if oi_then and oi_then.get("open_interest", 0) > 0:
                oi_change = (open_interest - oi_then["open_interest"]) / oi_then["open_interest"]
                if oi_change > 0.05:
                    oi_score = min(1.0, oi_change / 0.15)
                    scores.append(oi_score)
                    reasons.append(f"OI rising: +{oi_change:.1%}")

            if scores:
                component.raw_score = np.mean(scores)
                component.confidence = min(component.raw_score, 0.85)
                component.triggered = funding_rate > self.config.funding_high_threshold
                component.reason = "; ".join(reasons)
                component.data = {"funding_rate": funding_rate, "open_interest": open_interest}

        except Exception as e:
            logger.debug(f"Funding analysis error: {e}")

        return component

    def _analyze_liquidation_cascade(self, asset: str, now: datetime, derivatives_data: Optional[Dict]) -> ShortSignalComponent:
        """检测清算级联前兆"""
        component = ShortSignalComponent(
            signal_type=ShortSignalType.LIQUIDATION_CASCADE,
            raw_score=0.0, confidence=0.0, reason="", triggered=False,
        )
        if not derivatives_data:
            return component

        try:
            liq_long = derivatives_data.get("liquidations_long", 0)
            liq_short = derivatives_data.get("liquidations_short", 0)
            open_interest = derivatives_data.get("open_interest", 0)

            scores = []
            reasons = []

            # 1. Long liquidation spike (time-based)
            liq_window = self._get_lookback_series(
                self._liquidation_history[asset], now, self.config.liquidation_lookback_hours,
            )
            if len(liq_window) >= 3:
                avg_long_liq = np.mean([v["long"] for _, v in liq_window])
                if avg_long_liq > 0:
                    liq_ratio = liq_long / avg_long_liq
                    if liq_ratio > self.config.liquidation_long_spike_ratio:
                        liq_score = min(1.0, (liq_ratio - 1) / (self.config.liquidation_long_spike_ratio - 1))
                        scores.append(liq_score)
                        reasons.append(f"Long liquidations spike: {liq_ratio:.1f}x avg")

            # 2. Liquidation imbalance
            total_liq = liq_long + liq_short
            if total_liq > 0:
                imbalance = (liq_long - liq_short) / total_liq
                if imbalance > 0.6:
                    imbalance_score = min(1.0, (imbalance - 0.3) / 0.4)
                    scores.append(imbalance_score)
                    reasons.append(f"Liquidation imbalance: {imbalance:.1%} longs")

            # 3. OI at high percentile (time-based)
            all_oi_entries = self._get_lookback_series(
                self._liquidation_history[asset], now, self.config.volume_avg_window_hours,
            )
            if len(all_oi_entries) >= 10:
                oi_values = [v["oi"] for _, v in all_oi_entries if v["oi"] > 0]
                if oi_values:
                    oi_percentile = sum(1 for oi in oi_values if open_interest > oi) / len(oi_values)
                    if oi_percentile > 0.8:
                        oi_score = (oi_percentile - 0.8) / 0.2
                        scores.append(oi_score)
                        reasons.append(f"OI at {oi_percentile:.0%} percentile")

            if scores:
                component.raw_score = np.mean(scores)
                component.confidence = min(component.raw_score, 0.75)
                component.triggered = len(scores) >= 2
                component.reason = "; ".join(reasons)
                component.data = {"liquidations_long": liq_long, "liquidations_short": liq_short}

        except Exception as e:
            logger.debug(f"Liquidation analysis error: {e}")

        return component

    def _analyze_extreme_greed(self, asset: str, now: datetime, sentiment_data: Optional[Dict]) -> ShortSignalComponent:
        """检测极端贪婪情绪"""
        component = ShortSignalComponent(
            signal_type=ShortSignalType.EXTREME_GREED,
            raw_score=0.0, confidence=0.0, reason="", triggered=False,
        )
        if not sentiment_data:
            return component

        try:
            fg_raw = sentiment_data.get("fear_greed_index", 0.5)
            fear_greed, _ = self._normalize_fear_greed(fg_raw)
            social_sentiment = sentiment_data.get("social_sentiment", 0.0)

            scores = []
            reasons = []

            # 1. Fear/Greed extreme
            if fear_greed > self.config.greed_extreme_threshold:
                greed_score = min(1.0, (fear_greed - self.config.greed_extreme_threshold) / 0.15)
                scores.append(greed_score)
                reasons.append(f"Extreme greed: {fear_greed:.0%}")

            # 2. Social euphoria
            if social_sentiment > 0.7:
                social_score = min(1.0, (social_sentiment - 0.5) / 0.4)
                scores.append(social_score)
                reasons.append(f"Social euphoria: {social_sentiment:.0%}")

            # 3. Sustained high (time-based, P0)
            fg_window = self._get_lookback_series(
                self._sentiment_history[asset], now, self.config.fear_greed_lookback_hours,
            )
            if len(fg_window) >= 3:
                avg_fg = np.mean([v["fg"] for _, v in fg_window])
                if avg_fg > 0.7:
                    duration_score = min(1.0, (avg_fg - 0.6) / 0.2)
                    scores.append(duration_score)
                    reasons.append(f"Sustained greed: avg {avg_fg:.0%}")

            if scores:
                component.raw_score = np.mean(scores)
                component.confidence = min(component.raw_score, 0.70)
                component.triggered = fear_greed > self.config.greed_extreme_threshold
                component.reason = "; ".join(reasons)
                component.data = {"fear_greed": fear_greed, "social_sentiment": social_sentiment}

        except Exception as e:
            logger.debug(f"Sentiment analysis error: {e}")

        return component

    def _analyze_whale_outflow(self, onchain_data: Optional[Dict]) -> ShortSignalComponent:
        """检测鲸鱼大额流出 (stateless - no per-asset history needed)"""
        component = ShortSignalComponent(
            signal_type=ShortSignalType.WHALE_OUTFLOW,
            raw_score=0.0, confidence=0.0, reason="", triggered=False,
        )
        if not onchain_data:
            return component

        try:
            whale_flow = onchain_data.get("whale_flow_usd", 0)
            exchange_inflow = onchain_data.get("exchange_inflow", 0)

            scores = []
            reasons = []

            if whale_flow < -self.config.whale_outflow_threshold_usd:
                outflow_score = min(1.0, abs(whale_flow) / (self.config.whale_outflow_threshold_usd * 3))
                scores.append(outflow_score)
                reasons.append(f"Whale outflow: ${abs(whale_flow)/1e6:.0f}M")

            if exchange_inflow > self.config.whale_outflow_threshold_usd:
                inflow_score = min(1.0, exchange_inflow / (self.config.whale_outflow_threshold_usd * 2))
                scores.append(inflow_score)
                reasons.append(f"Exchange inflow: ${exchange_inflow/1e6:.0f}M")

            if scores:
                component.raw_score = np.mean(scores)
                component.confidence = min(component.raw_score, 0.65)
                component.triggered = len(scores) >= 1
                component.reason = "; ".join(reasons)
                component.data = {"whale_flow_usd": whale_flow, "exchange_inflow": exchange_inflow}

        except Exception as e:
            logger.debug(f"Whale analysis error: {e}")

        return component

    def _analyze_structure_break(self, asset: str, now: datetime, market_data: Dict) -> ShortSignalComponent:
        """[UPG2] 检测支撑位跌破 - 空头核心 alpha"""
        component = ShortSignalComponent(
            signal_type=ShortSignalType.STRUCTURE_BREAK,
            raw_score=0.0, confidence=0.0, reason="", triggered=False,
        )
        try:
            price = market_data.get("price", 0)
            if price <= 0:
                return component

            lookback_entries = self._get_lookback_series(
                self._price_history[asset], now,
                self.config.structure_break_lookback_hours,
            )

            if len(lookback_entries) < self.config.structure_break_min_bars:
                return component

            prices = [v for _, v in lookback_entries]

            # Exclude last 6 bars (24h) - find support from older history
            if len(prices) <= 6:
                return component
            historical_prices = prices[:-6]

            support = min(historical_prices)
            if support <= 0:
                return component

            # Break magnitude: positive = price below support
            break_magnitude = (support - price) / support

            scores = []
            reasons = []

            if break_magnitude > self.config.structure_break_threshold:
                # 1. Break magnitude score (5% = full score)
                break_score = min(1.0, break_magnitude / 0.05)
                scores.append(break_score)
                reasons.append(f"Support break: {break_magnitude:.1%} below {support:.0f}")

                # 2. Volume confirmation
                volume = market_data.get("volume", 0)
                vol_entries = self._get_lookback_series(
                    self._volume_history[asset], now,
                    self.config.volume_avg_window_hours,
                )
                if len(vol_entries) >= 5 and volume > 0:
                    avg_vol = np.mean([v for _, v in vol_entries])
                    if avg_vol > 0:
                        vol_ratio = volume / avg_vol
                        if vol_ratio > 1.5:
                            vol_score = min(1.0, (vol_ratio - 1.0) / 2.0)
                            scores.append(vol_score)
                            reasons.append(f"Break volume: {vol_ratio:.1f}x avg")

                # 3. Sustained break (last 3 bars all below support)
                recent = prices[-3:]
                if all(p < support for p in recent):
                    scores.append(0.6)
                    reasons.append("Sustained break (3 bars below support)")

            if scores:
                component.raw_score = np.mean(scores)
                component.confidence = min(component.raw_score, 0.80)
                component.triggered = len(scores) >= 1
                component.reason = "; ".join(reasons)
                component.data = {
                    "support_level": support,
                    "break_magnitude": break_magnitude,
                    "price": price,
                }

        except Exception as e:
            logger.debug(f"[UPG2] Structure break analysis error: {e}")

        return component

    # =========================================================================
    # SIGNAL SYNTHESIS
    # =========================================================================

    def _synthesize_signal(
        self,
        asset: str,
        now: datetime,
        components: List[ShortSignalComponent],
        meta: Dict,
    ) -> ShortBiasSignal:
        """综合各组件信号 - direction only -1 or 0."""
        signal = ShortBiasSignal(timestamp=now, asset=asset)
        signal.components = components

        weights = [
            self.config.weight_parabolic,
            self.config.weight_funding,
            self.config.weight_liquidation,
            self.config.weight_sentiment,
            self.config.weight_whale,
            self.config.weight_structure_break,  # [UPG2]
        ]

        triggered_count = 0
        weighted_scores = []
        triggered_weights = []
        reasons = []
        component_scores: Dict[str, Dict] = {}

        for comp, weight in zip(components, weights):
            component_scores[comp.signal_type.value] = {
                "raw_score": round(comp.raw_score, 4),
                "confidence": round(comp.confidence, 4),
                "triggered": comp.triggered,
            }
            if comp.triggered:
                triggered_count += 1
                weighted_scores.append(comp.confidence * weight)
                triggered_weights.append(weight)
                reasons.append(f"[{comp.signal_type.value}] {comp.reason}")

        # Diagnostics
        total_possible = len(components) - len(meta.get("missing_inputs", []))
        meta["components"] = component_scores
        meta["triggered_components"] = [
            c.signal_type.value for c in components if c.triggered
        ]
        meta["coverage_ratio"] = round(total_possible / len(components), 2) if components else 0

        if not weighted_scores:
            signal.meta = meta
            return signal

        # Coverage-aware fusion: renormalize over triggered weights
        total_weight = sum(triggered_weights)
        raw_confidence = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        # Multi-confirmation bonus
        if triggered_count >= 3:
            raw_confidence *= 1.15
        elif triggered_count >= 2:
            raw_confidence *= 1.05

        final_confidence = min(raw_confidence, self.config.max_confidence)

        if final_confidence >= self.config.min_confidence_threshold:
            signal.direction = -1.0  # ONLY short
            signal.confidence = final_confidence
            signal.reasons = reasons

            max_comp = max(components, key=lambda c: c.confidence if c.triggered else 0)
            signal.primary_trigger = max_comp.signal_type if max_comp.triggered else ShortSignalType.COMPOSITE

            if final_confidence > 0.7:
                signal.suggested_stop_pct = 0.025
                signal.max_exposure_pct = 0.35
            elif final_confidence > 0.5:
                signal.suggested_stop_pct = 0.03
                signal.max_exposure_pct = 0.30
            else:
                signal.suggested_stop_pct = 0.04
                signal.max_exposure_pct = 0.20

        signal.meta = meta
        return signal

    # =========================================================================
    # FUNDING ARB (short-only: boost short confidence, never produce longs)
    # =========================================================================

    def _apply_funding_arb(self, signal: ShortBiasSignal, derivatives_data: Optional[Dict]):
        """
        v6.7 Funding Rate Arbitrage - short-only.
        Extreme positive funding -> boost short confidence.
        Negative funding note recorded as metadata (no long output).
        """
        if not derivatives_data:
            return

        funding = derivatives_data.get("funding_rate", 0.0)

        if signal.direction < 0 and funding > self.config.funding_arb_short_boost_threshold:
            boost = self.config.funding_arb_short_confidence_boost
            signal.confidence = min(signal.confidence + boost, self.config.max_confidence)
            signal.reasons.append(
                f"[FUNDING_ARB] Extreme funding {funding:.4%} -> short boost +{boost:.0%}"
            )
            signal.meta.setdefault("funding_arb", {})["short_boost"] = boost
        elif funding < -0.0016:
            # Record as metadata only - short-only agent never produces positive direction
            signal.meta.setdefault("funding_arb", {})["note"] = (
                f"Negative funding {funding:.4%} favors longs (not applicable: short-only agent)"
            )

    # =========================================================================
    # REGIME GATING (P2, optional)
    # =========================================================================

    def _apply_regime_gating(self, signal: ShortBiasSignal, regime_state: str, regime_confidence: float):
        """Penalize confidence in quiet/consolidation regimes."""
        regime_upper = regime_state.upper()
        if any(q in regime_upper for q in self.config.regime_quiet_names):
            penalty = self.config.regime_quiet_penalty
            signal.confidence *= penalty
            signal.reasons.append(
                f"[REGIME] {regime_state} (conf={regime_confidence:.2f}) -> "
                f"confidence ×{penalty}"
            )
            signal.meta.setdefault("regime_gating", {}).update({
                "regime": regime_state,
                "regime_confidence": regime_confidence,
                "penalty_applied": penalty,
            })
            # [FIX-DA4] Re-validate threshold after regime penalty.
            # Without this, direction=-1.0 persists with sub-threshold confidence,
            # causing is_actionable()=True on invalid signals.
            if signal.confidence < self.config.min_confidence_threshold:
                signal.direction = 0.0
                signal.reasons.append(
                    f"[REGIME_GATE] confidence {signal.confidence:.3f} below "
                    f"min_threshold {self.config.min_confidence_threshold} after regime penalty -> neutralized"
                )

    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================

    def get_stats(self, asset: Optional[str] = None) -> Dict:
        """获取 Agent 统计信息. If asset=None, return combined."""
        if asset:
            return self._get_asset_stats(asset)

        all_assets = sorted(set(
            list(self._total_signals.keys()) +
            list(self._consecutive_signals.keys())
        ))
        return {
            "total_signals": sum(self._total_signals.get(a, 0) for a in all_assets),
            "per_asset": {a: self._get_asset_stats(a) for a in all_assets},
            "config": {
                "max_confidence": self.config.max_confidence,
                "min_confidence_threshold": self.config.min_confidence_threshold,
                "signal_cooldown_minutes": self.config.signal_cooldown_minutes,
            },
        }

    def _get_asset_stats(self, asset: str) -> Dict:
        return {
            "total_signals": self._total_signals.get(asset, 0),
            "consecutive_signals": self._consecutive_signals.get(asset, 0),
            "signal_breakdown": {
                k.value: v for k, v in self._signal_breakdown.get(asset, {}).items()
            },
            "observation_minutes": round(
                (datetime.now(timezone.utc) - self._observation_start.get(asset, datetime.now(timezone.utc))).total_seconds() / 60, 1
            ),
            "history_lengths": {
                "price": len(self._price_history.get(asset, [])),
                "volume": len(self._volume_history.get(asset, [])),
                "funding": len(self._funding_history.get(asset, [])),
                "liquidation": len(self._liquidation_history.get(asset, [])),
                "sentiment": len(self._sentiment_history.get(asset, [])),
            },
        }

    def reset_state(self, asset: Optional[str] = None):
        """重置 Agent 状态. If asset=None, reset all."""
        if asset:
            self._last_signal_time.pop(asset, None)
            self._consecutive_signals[asset] = 0
        else:
            self._last_signal_time.clear()
            for a in self._consecutive_signals:
                self._consecutive_signals[a] = 0
        logger.info(f"[SHORT_BIAS] State reset (asset={asset or 'ALL'})")


# =============================================================================
# SINGLETON
# =============================================================================

_short_bias_agent: Optional[ShortBiasAgent] = None


def get_short_bias_agent(config: ShortBiasConfig = None) -> ShortBiasAgent:
    """获取或创建单例实例"""
    global _short_bias_agent
    if _short_bias_agent is None:
        _short_bias_agent = ShortBiasAgent(config)
    elif config is not None and _short_bias_agent.config != config:
        logger.info("[SHORT_BIAS] Config changed; refreshing singleton instance")
        _short_bias_agent = ShortBiasAgent(config)
    return _short_bias_agent


def reset_short_bias_agent():
    """重置单例"""
    global _short_bias_agent
    _short_bias_agent = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ShortBiasConfig',
    'ShortSignalType',
    'ShortSignalComponent',
    'ShortBiasSignal',
    'ShortBiasAgent',
    'get_short_bias_agent',
    'reset_short_bias_agent',
]


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SHORT-BIASED AGENT TEST (v2.0)")
    print("=" * 70)
    print()

    test_config = ShortBiasConfig(min_observation_minutes=0)
    agent = ShortBiasAgent(config=test_config)

    market_data = {
        "asset": "BTC",
        "price": 200.0,
        "high_24h": 210.0,
        "low_24h": 180.0,
        "volume": 5000000,
        "rsi": 85.0,
    }
    derivatives_data = {
        "funding_rate": 0.0012,
        "open_interest": 2_000_000_000,
        "liquidations_long": 50_000_000,
        "liquidations_short": 5_000_000,
    }
    sentiment_data = {"fear_greed_index": 0.90, "social_sentiment": 0.85}
    onchain_data = {"whale_flow_usd": -100_000_000, "exchange_inflow": 80_000_000}

    print("[Test 1] Extreme greed market (should trigger short)")
    signal = agent.analyze(market_data, derivatives_data, sentiment_data, onchain_data)
    print(f"  Direction: {signal.direction}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Asset: {signal.asset}")
    print(f"  Primary Trigger: {signal.primary_trigger.value}")
    print(f"  Meta keys: {list(signal.meta.keys())}")
    assert signal.direction <= 0, "CRITICAL: Agent attempted to go LONG!"
    print("  OK: direction <= 0")

    d = signal.to_agent_signal_dict()
    assert "meta" in d
    assert "asset" in d
    print(f"  to_agent_signal_dict has meta: {bool(d['meta'])}")

    print()
    print("[Test 2] Normal market")
    normal = {"asset": "ETH", "price": 100.0, "high_24h": 102.0, "low_24h": 98.0, "volume": 1e6, "rsi": 50}
    s2 = agent.analyze(normal, {"funding_rate": 0.0001, "open_interest": 1e9, "liquidations_long": 5e6, "liquidations_short": 5e6},
                        {"fear_greed_index": 50}, asset="ETH")
    print(f"  Direction: {s2.direction}, Actionable: {s2.is_actionable()}")
    assert s2.direction <= 0

    print()
    print(f"Stats: {agent.get_stats()}")
    print()
    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)

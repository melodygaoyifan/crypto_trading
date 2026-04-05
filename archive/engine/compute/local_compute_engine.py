#!/usr/bin/env python3
"""
LocalComputeEngine - The Edge Brain
====================================
SOTA v3.0 Component for RTX 5090 (32GB VRAM)

Capabilities:
1. GPU-accelerated VPIN (Volume-synchronized Probability of Informed Trading)
2. GPU-accelerated OFI (Order Flow Imbalance)
3. Local SLM (Llama-3-8B) for Flash Sentiment Analysis
4. 4σ Escalation Logic to Master Agent

Hardware: RTX 5090 (32GB VRAM)
Frequency: Real-time microstructure, 4H/1D strategy

Author: Senior Quant Team
Version: SOTA v3.0
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import queue

import numpy as np

# Conditional imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LocalComputeEngine')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TradeDirection(Enum):
    """Trade direction classification."""
    BUY = 1
    SELL = -1
    UNKNOWN = 0


@dataclass
class Trade:
    """Single trade record."""
    timestamp: float
    price: float
    volume: float
    direction: TradeDirection
    asset: str = 'BTC'


@dataclass
class OrderBookSnapshot:
    """L2 orderbook snapshot."""
    timestamp: float
    asset: str
    bids: np.ndarray  # [[price, size], ...]
    asks: np.ndarray  # [[price, size], ...]
    
    @property
    def mid_price(self) -> float:
        if len(self.bids) > 0 and len(self.asks) > 0:
            return (self.bids[0, 0] + self.asks[0, 0]) / 2
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        if len(self.bids) > 0 and len(self.asks) > 0:
            return (self.asks[0, 0] - self.bids[0, 0]) / self.mid_price * 10000
        return 0.0


@dataclass
class SentimentSignal:
    """Sentiment analysis result."""
    timestamp: float
    source: str  # 'twitter', 'news', 'reddit'
    text: str
    sentiment_score: float  # [-1, +1]
    confidence: float  # [0, 1]
    sigma_deviation: float  # Standard deviations from mean
    asset_mentions: Dict[str, int] = field(default_factory=dict)
    escalate: bool = False  # True if >4σ shock


@dataclass
class MicrostructureMetrics:
    """Combined microstructure metrics."""
    timestamp: float
    asset: str
    vpin: float  # [0, 1] - probability of informed trading
    ofi: float  # Order flow imbalance
    ofi_cumulative: float  # Cumulative OFI
    toxicity_level: str  # 'LOW', 'MODERATE', 'HIGH', 'TOXIC'
    recommended_mode: str  # 'AGGRESSIVE', 'PASSIVE', 'HIDDEN', 'ABORT'


@dataclass
class LocalComputeResult:
    """Complete output from LocalComputeEngine."""
    timestamp: float
    microstructure: Dict[str, MicrostructureMetrics]
    sentiment: Optional[SentimentSignal]
    escalation_required: bool
    escalation_reason: Optional[str]
    latency_ms: float


# =============================================================================
# GPU-ACCELERATED VPIN CALCULATOR
# =============================================================================

class GPUVPINCalculator:
    """
    Volume-synchronized Probability of Informed Trading (VPIN)
    
    VPIN measures the probability that trading is driven by informed traders.
    High VPIN -> High toxicity -> Avoid aggressive execution.
    
    Formula:
        VPIN = Σ|V_buy - V_sell| / (n × V_bucket)
        
    Where:
        - V_buy, V_sell = Volume classified as buys/sells
        - V_bucket = Volume bucket size
        - n = Number of buckets
    
    GPU Acceleration: Process trade classification and bucketing on RTX 5090.
    """
    
    def __init__(
        self,
        bucket_size: float = 50000.0,  # USD volume per bucket
        n_buckets: int = 50,  # Rolling window of buckets
        use_gpu: bool = True,
    ):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # GPU/CPU array module
        self.xp = cp if self.use_gpu else np
        
        # State
        self.buckets: deque = deque(maxlen=n_buckets)
        self.current_bucket_volume = 0.0
        self.current_bucket_buy = 0.0
        self.current_bucket_sell = 0.0
        
        # Trade buffer for GPU batch processing
        self.trade_buffer: List[Trade] = []
        self.buffer_size = 1000  # Process in batches
        
        # Price history for tick rule
        self.last_prices: deque = deque(maxlen=100)
        
        logger.info(f"VPIN Calculator initialized | GPU: {self.use_gpu} | "
                   f"Bucket: ${bucket_size:,.0f} | Windows: {n_buckets}")
    
    def classify_trade_tick_rule(self, price: float) -> TradeDirection:
        """
        Classify trade direction using tick rule.
        
        Tick Rule:
        - If price > last_price -> BUY
        - If price < last_price -> SELL
        - If price == last_price -> Use previous classification
        """
        if len(self.last_prices) == 0:
            return TradeDirection.UNKNOWN
        
        last_price = self.last_prices[-1]
        
        if price > last_price:
            return TradeDirection.BUY
        elif price < last_price:
            return TradeDirection.SELL
        else:
            # Look back for last different price
            for p in reversed(self.last_prices):
                if p != price:
                    return TradeDirection.BUY if price > p else TradeDirection.SELL
            return TradeDirection.UNKNOWN
    
    def _classify_trades_gpu(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated trade classification using bulk tick rule.
        
        Returns:
            buy_volumes, sell_volumes
        """
        # Transfer to GPU
        prices_gpu = self.xp.asarray(prices)
        volumes_gpu = self.xp.asarray(volumes)
        
        # Calculate price changes
        price_changes = self.xp.diff(prices_gpu, prepend=prices_gpu[0])
        
        # Classify: positive change = buy, negative = sell
        buy_mask = price_changes > 0
        sell_mask = price_changes < 0
        
        # Handle zero changes (use previous classification)
        zero_mask = price_changes == 0
        if self.xp.any(zero_mask):
            # Forward fill the last non-zero classification
            for i in range(1, len(price_changes)):
                if zero_mask[i]:
                    buy_mask[i] = buy_mask[i - 1]
                    sell_mask[i] = sell_mask[i - 1]
        
        buy_volumes = self.xp.where(buy_mask, volumes_gpu, 0.0)
        sell_volumes = self.xp.where(sell_mask, volumes_gpu, 0.0)
        
        # Transfer back to CPU if needed
        if self.use_gpu:
            buy_volumes = cp.asnumpy(buy_volumes)
            sell_volumes = cp.asnumpy(sell_volumes)
        
        return buy_volumes, sell_volumes
    
    def add_trade(self, trade: Trade) -> Optional[float]:
        """
        Add a trade and return VPIN if bucket completed.
        """
        # Update price history
        self.last_prices.append(trade.price)
        
        # Classify trade
        if trade.direction == TradeDirection.UNKNOWN:
            direction = self.classify_trade_tick_rule(trade.price)
        else:
            direction = trade.direction
        
        # Add to current bucket
        trade_value = trade.price * trade.volume
        self.current_bucket_volume += trade_value
        
        if direction == TradeDirection.BUY:
            self.current_bucket_buy += trade_value
        elif direction == TradeDirection.SELL:
            self.current_bucket_sell += trade_value
        else:
            # Split unknown 50/50
            self.current_bucket_buy += trade_value / 2
            self.current_bucket_sell += trade_value / 2
        
        # Check if bucket is full
        if self.current_bucket_volume >= self.bucket_size:
            # Complete bucket
            imbalance = abs(self.current_bucket_buy - self.current_bucket_sell)
            self.buckets.append(imbalance)
            
            # Reset current bucket
            self.current_bucket_volume = 0.0
            self.current_bucket_buy = 0.0
            self.current_bucket_sell = 0.0
            
            # Calculate VPIN
            return self.calculate_vpin()
        
        return None
    
    def add_trades_batch(self, trades: List[Trade]) -> Optional[float]:
        """
        GPU-accelerated batch trade processing.
        """
        if len(trades) == 0:
            return None
        
        # Extract arrays
        prices = np.array([t.price for t in trades])
        volumes = np.array([t.volume for t in trades])
        
        # GPU classification
        buy_volumes, sell_volumes = self._classify_trades_gpu(prices, volumes)
        
        # Process into buckets
        for i, trade in enumerate(trades):
            self.last_prices.append(trade.price)
            
            trade_value = trade.price * trade.volume
            self.current_bucket_volume += trade_value
            self.current_bucket_buy += buy_volumes[i] * trade.price
            self.current_bucket_sell += sell_volumes[i] * trade.price
            
            if self.current_bucket_volume >= self.bucket_size:
                imbalance = abs(self.current_bucket_buy - self.current_bucket_sell)
                self.buckets.append(imbalance)
                self.current_bucket_volume = 0.0
                self.current_bucket_buy = 0.0
                self.current_bucket_sell = 0.0
        
        return self.calculate_vpin()
    
    def calculate_vpin(self) -> float:
        """
        Calculate current VPIN value.
        
        VPIN = Σ|V_buy - V_sell| / (n × V_bucket)
        """
        if len(self.buckets) < 2:
            return 0.5  # Default to neutral
        
        total_imbalance = sum(self.buckets)
        vpin = total_imbalance / (len(self.buckets) * self.bucket_size)
        
        # Clamp to [0, 1]
        return min(max(vpin, 0.0), 1.0)
    
    def get_toxicity_level(self, vpin: float) -> str:
        """
        Convert VPIN to toxicity level.
        
        Thresholds calibrated for crypto markets:
        - LOW: VPIN < 0.3
        - MODERATE: 0.3 <= VPIN < 0.5
        - HIGH: 0.5 <= VPIN < 0.7
        - TOXIC: VPIN >= 0.7
        """
        if vpin < 0.3:
            return 'LOW'
        elif vpin < 0.5:
            return 'MODERATE'
        elif vpin < 0.7:
            return 'HIGH'
        else:
            return 'TOXIC'


# =============================================================================
# GPU-ACCELERATED OFI CALCULATOR
# =============================================================================

class GPUOFICalculator:
    """
    Order Flow Imbalance (OFI) Calculator
    
    OFI measures the imbalance in order flow at the best bid/ask.
    Positive OFI -> Buy pressure -> Price likely to increase.
    
    Formula:
        OFI_t = ΔQ_bid × I(P_bid ≥ P_bid,t-1) - ΔQ_ask × I(P_ask ≤ P_ask,t-1)
        
    Where:
        - ΔQ = Change in quantity at best level
        - I() = Indicator function
        - P = Price at level
    
    GPU Acceleration: Process orderbook updates in parallel.
    """
    
    def __init__(
        self,
        lookback_periods: int = 100,
        use_gpu: bool = True,
    ):
        self.lookback_periods = lookback_periods
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # State per asset
        self.last_book: Dict[str, OrderBookSnapshot] = {}
        self.ofi_history: Dict[str, deque] = {}
        self.cumulative_ofi: Dict[str, float] = {}
        
        logger.info(f"OFI Calculator initialized | GPU: {self.use_gpu} | "
                   f"Lookback: {lookback_periods}")
    
    def _init_asset(self, asset: str):
        """Initialize state for an asset."""
        if asset not in self.ofi_history:
            self.ofi_history[asset] = deque(maxlen=self.lookback_periods)
            self.cumulative_ofi[asset] = 0.0
    
    def update(self, book: OrderBookSnapshot) -> float:
        """
        Update OFI with new orderbook snapshot.
        
        Returns:
            Current OFI value
        """
        asset = book.asset
        self._init_asset(asset)
        
        # Get previous book
        last = self.last_book.get(asset)
        
        if last is None:
            self.last_book[asset] = book
            return 0.0
        
        # Calculate OFI
        ofi = self._calculate_ofi(last, book)
        
        # Update state
        self.ofi_history[asset].append(ofi)
        self.cumulative_ofi[asset] += ofi
        self.last_book[asset] = book
        
        return ofi
    
    def _calculate_ofi(
        self,
        prev: OrderBookSnapshot,
        curr: OrderBookSnapshot,
    ) -> float:
        """
        Calculate single-period OFI.
        
        OFI = Δbid_size × I(bid_price_up) - Δask_size × I(ask_price_down)
        """
        # Best bid/ask from previous
        prev_bid_price = prev.bids[0, 0] if len(prev.bids) > 0 else 0
        prev_bid_size = prev.bids[0, 1] if len(prev.bids) > 0 else 0
        prev_ask_price = prev.asks[0, 0] if len(prev.asks) > 0 else float('inf')
        prev_ask_size = prev.asks[0, 1] if len(prev.asks) > 0 else 0
        
        # Best bid/ask from current
        curr_bid_price = curr.bids[0, 0] if len(curr.bids) > 0 else 0
        curr_bid_size = curr.bids[0, 1] if len(curr.bids) > 0 else 0
        curr_ask_price = curr.asks[0, 0] if len(curr.asks) > 0 else float('inf')
        curr_ask_size = curr.asks[0, 1] if len(curr.asks) > 0 else 0
        
        # OFI components
        ofi = 0.0
        
        # Bid side contribution
        if curr_bid_price >= prev_bid_price:
            # Bid improved or held - add size change
            if curr_bid_price > prev_bid_price:
                ofi += curr_bid_size  # New level, full size
            else:
                ofi += curr_bid_size - prev_bid_size
        else:
            # Bid retreated - negative contribution
            ofi -= prev_bid_size
        
        # Ask side contribution
        if curr_ask_price <= prev_ask_price:
            # Ask improved or held - subtract size change
            if curr_ask_price < prev_ask_price:
                ofi -= curr_ask_size  # New level, full size negative
            else:
                ofi -= (curr_ask_size - prev_ask_size)
        else:
            # Ask retreated - positive contribution
            ofi += prev_ask_size
        
        return ofi
    
    def calculate_ofi_batch_gpu(
        self,
        books: List[OrderBookSnapshot],
    ) -> np.ndarray:
        """
        GPU-accelerated batch OFI calculation.
        
        Process multiple orderbook snapshots in parallel.
        """
        if len(books) < 2:
            return np.array([0.0])
        
        # Extract arrays
        bid_prices = np.array([b.bids[0, 0] if len(b.bids) > 0 else 0 for b in books])
        bid_sizes = np.array([b.bids[0, 1] if len(b.bids) > 0 else 0 for b in books])
        ask_prices = np.array([b.asks[0, 0] if len(b.asks) > 0 else 0 for b in books])
        ask_sizes = np.array([b.asks[0, 1] if len(b.asks) > 0 else 0 for b in books])
        
        # Transfer to GPU
        bp = self.xp.asarray(bid_prices)
        bs = self.xp.asarray(bid_sizes)
        ap = self.xp.asarray(ask_prices)
        az = self.xp.asarray(ask_sizes)
        
        # Calculate changes
        bp_prev = self.xp.roll(bp, 1)
        bs_prev = self.xp.roll(bs, 1)
        ap_prev = self.xp.roll(ap, 1)
        az_prev = self.xp.roll(az, 1)
        
        # OFI calculation (vectorized)
        bid_improved = bp >= bp_prev
        bid_new_level = bp > bp_prev
        
        ofi_bid = self.xp.where(
            bid_improved,
            self.xp.where(bid_new_level, bs, bs - bs_prev),
            -bs_prev
        )
        
        ask_improved = ap <= ap_prev
        ask_new_level = ap < ap_prev
        
        ofi_ask = self.xp.where(
            ask_improved,
            self.xp.where(ask_new_level, -az, -(az - az_prev)),
            az_prev
        )
        
        ofi = ofi_bid + ofi_ask
        ofi[0] = 0  # First element invalid
        
        # Transfer back
        if self.use_gpu:
            ofi = cp.asnumpy(ofi)
        
        return ofi
    
    def get_ofi_stats(self, asset: str) -> Dict[str, float]:
        """Get OFI statistics for an asset."""
        self._init_asset(asset)
        
        history = list(self.ofi_history[asset])
        if len(history) == 0:
            return {
                'current': 0.0,
                'cumulative': 0.0,
                'mean': 0.0,
                'std': 0.0,
                'zscore': 0.0,
            }
        
        current = history[-1] if history else 0.0
        mean = np.mean(history)
        std = np.std(history) if len(history) > 1 else 1.0
        
        return {
            'current': current,
            'cumulative': self.cumulative_ofi[asset],
            'mean': mean,
            'std': std,
            'zscore': (current - mean) / std if std > 0 else 0.0,
        }


# =============================================================================
# LOCAL SLM SENTIMENT ANALYZER
# =============================================================================

class LocalSLMSentimentAnalyzer:
    """
    Local Small Language Model for Flash Sentiment Analysis
    
    Uses Llama-3-8B (or similar) running on RTX 5090 for:
    1. Real-time sentiment scoring of Twitter/News streams
    2. Asset mention extraction
    3. 4σ shock detection with escalation logic
    
    Only escalates >4σ sentiment shocks to Master Agent API.
    """
    
    # Sentiment analysis prompt template
    SENTIMENT_PROMPT = """Analyze the following crypto-related text and provide:
1. Sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish)
2. Confidence from 0.0 to 1.0
3. Assets mentioned (BTC, ETH, SOL)

Text: "{text}"

Respond in JSON format:
{{"sentiment": <float>, "confidence": <float>, "assets": ["BTC", "ETH", "SOL"]}}"""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda",
        max_batch_size: int = 8,
        escalation_sigma: float = 4.0,
        sentiment_window: int = 1000,
    ):
        self.model_name = model_name
        self.device = device
        self.max_batch_size = max_batch_size
        self.escalation_sigma = escalation_sigma
        
        # Sentiment history for σ calculation
        self.sentiment_history = deque(maxlen=sentiment_window)
        
        # Model (lazy loaded)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False
        
        # Processing queue
        self.text_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.mean_sentiment = 0.0
        self.std_sentiment = 0.1  # Initial estimate
        
        logger.info(f"Local SLM Analyzer initialized | Model: {model_name} | "
                   f"Escalation: {escalation_sigma}σ")
    
    def load_model(self):
        """Load the SLM model onto GPU."""
        if self.model_loaded:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available - using mock sentiment")
            self.model_loaded = True
            return
        
        try:
            logger.info(f"Loading model {self.model_name} to {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=100,
                do_sample=False,
            )
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Falling back to mock sentiment analysis")
            self.model_loaded = True  # Mark as loaded to use mock
    
    def _mock_sentiment(self, text: str) -> Dict:
        """Mock sentiment analysis when model unavailable."""
        # Simple keyword-based mock
        bullish_keywords = ['bull', 'moon', 'pump', 'buy', 'long', 'breakout', 'ath']
        bearish_keywords = ['bear', 'dump', 'sell', 'short', 'crash', 'rekt', 'liquidat']
        
        text_lower = text.lower()
        
        bull_count = sum(1 for kw in bullish_keywords if kw in text_lower)
        bear_count = sum(1 for kw in bearish_keywords if kw in text_lower)
        
        if bull_count + bear_count == 0:
            sentiment = 0.0
            confidence = 0.3
        else:
            sentiment = (bull_count - bear_count) / (bull_count + bear_count)
            confidence = min(0.9, 0.3 + 0.1 * (bull_count + bear_count))
        
        # Extract asset mentions
        assets = []
        if 'btc' in text_lower or 'bitcoin' in text_lower:
            assets.append('BTC')
        if 'eth' in text_lower or 'ethereum' in text_lower:
            assets.append('ETH')
        if 'sol' in text_lower or 'solana' in text_lower:
            assets.append('SOL')
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'assets': assets if assets else ['BTC', 'ETH', 'SOL'],
        }
    
    def _run_inference(self, text: str) -> Dict:
        """Run model inference for sentiment analysis."""
        if self.pipeline is None:
            return self._mock_sentiment(text)
        
        try:
            prompt = self.SENTIMENT_PROMPT.format(text=text[:500])  # Truncate long texts
            
            result = self.pipeline(prompt)[0]['generated_text']
            
            # Extract JSON from response
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(result[json_start:json_end])
                return {
                    'sentiment': float(parsed.get('sentiment', 0.0)),
                    'confidence': float(parsed.get('confidence', 0.5)),
                    'assets': parsed.get('assets', ['BTC', 'ETH', 'SOL']),
                }
            
            return self._mock_sentiment(text)
            
        except Exception as e:
            logger.debug(f"Inference error: {e}")
            return self._mock_sentiment(text)
    
    def analyze(self, text: str, source: str = 'unknown') -> SentimentSignal:
        """
        Analyze sentiment of a single text.
        
        Returns:
            SentimentSignal with escalation flag if >4σ
        """
        if not self.model_loaded:
            self.load_model()
        
        # Run inference
        result = self._run_inference(text)
        
        sentiment = result['sentiment']
        confidence = result['confidence']
        
        # Update statistics
        self.sentiment_history.append(sentiment)
        
        if len(self.sentiment_history) > 10:
            self.mean_sentiment = np.mean(self.sentiment_history)
            self.std_sentiment = max(np.std(self.sentiment_history), 0.01)
        
        # Calculate σ deviation
        sigma_deviation = (sentiment - self.mean_sentiment) / self.std_sentiment
        
        # Check for escalation
        escalate = abs(sigma_deviation) >= self.escalation_sigma
        
        return SentimentSignal(
            timestamp=time.time(),
            source=source,
            text=text[:200],  # Truncate for storage
            sentiment_score=sentiment,
            confidence=confidence,
            sigma_deviation=sigma_deviation,
            asset_mentions={asset: 1 for asset in result['assets']},
            escalate=escalate,
        )
    
    def analyze_batch(self, texts: List[Tuple[str, str]]) -> List[SentimentSignal]:
        """
        Analyze batch of (text, source) tuples.
        """
        return [self.analyze(text, source) for text, source in texts]
    
    def get_aggregated_sentiment(self, window_minutes: int = 60) -> Dict[str, float]:
        """
        Get aggregated sentiment over recent window.
        """
        recent = list(self.sentiment_history)[-window_minutes:]
        
        if len(recent) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'current_sigma': 0.0,
                'trend': 0.0,
            }
        
        return {
            'mean': np.mean(recent),
            'std': np.std(recent) if len(recent) > 1 else 0.0,
            'current_sigma': (recent[-1] - self.mean_sentiment) / self.std_sentiment if recent else 0.0,
            'trend': np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) > 1 else 0.0,
        }


# =============================================================================
# SOLANA NETWORK STRESS MONITOR
# =============================================================================

class SolanaNetworkStressMonitor:
    """
    Monitor Solana Network Stress Metrics
    
    Integrates:
    1. Compute Unit (CU) utilization
    2. Priority fees (tips)
    3. TPS (transactions per second)
    4. Slot skip rate
    
    Used for SOL-specific regime detection and execution decisions.
    """
    
    # Stress level thresholds
    CU_THRESHOLDS = {
        'LOW': 0.50,        # < 50% utilization
        'MODERATE': 0.70,   # 50-70%
        'HIGH': 0.85,       # 70-85%
        'CRITICAL': 0.95,   # 85-95%
        'CONGESTED': 1.00,  # > 95%
    }
    
    PRIORITY_FEE_THRESHOLDS = {
        'LOW': 1000,        # < 1000 microlamports
        'MODERATE': 10000,  # 1k - 10k
        'HIGH': 100000,     # 10k - 100k
        'EXTREME': 1000000, # > 100k
    }
    
    def __init__(
        self,
        history_size: int = 100,
    ):
        self.history_size = history_size
        
        # Metric histories
        self.cu_history = deque(maxlen=history_size)
        self.priority_fee_history = deque(maxlen=history_size)
        self.tps_history = deque(maxlen=history_size)
        self.slot_skip_history = deque(maxlen=history_size)
        
        # Current state
        self.current_stress_level = 'LOW'
        self.last_update = 0.0
        
        logger.info("Solana Network Stress Monitor initialized")
    
    def update(
        self,
        cu_used: float,
        cu_limit: float,
        priority_fee: float,
        tps: float,
        slot_skip_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Update with latest network metrics.
        
        Args:
            cu_used: Compute units used in recent blocks
            cu_limit: Maximum compute units per block
            priority_fee: Median priority fee in microlamports
            tps: Current transactions per second
            slot_skip_rate: Percentage of skipped slots
        """
        self.last_update = time.time()
        
        # Calculate utilization
        cu_utilization = cu_used / cu_limit if cu_limit > 0 else 0.0
        
        # Update histories
        self.cu_history.append(cu_utilization)
        self.priority_fee_history.append(priority_fee)
        self.tps_history.append(tps)
        self.slot_skip_history.append(slot_skip_rate)
        
        # Determine stress level
        stress_score = self._calculate_stress_score(
            cu_utilization, priority_fee, tps, slot_skip_rate
        )
        
        self.current_stress_level = self._score_to_level(stress_score)
        
        return {
            'cu_utilization': cu_utilization,
            'priority_fee': priority_fee,
            'tps': tps,
            'slot_skip_rate': slot_skip_rate,
            'stress_score': stress_score,
            'stress_level': self.current_stress_level,
            'execution_recommendation': self._get_execution_recommendation(),
        }
    
    def _calculate_stress_score(
        self,
        cu_util: float,
        priority_fee: float,
        tps: float,
        slot_skip: float,
    ) -> float:
        """
        Calculate composite stress score [0, 1].
        
        Weights:
        - CU utilization: 40%
        - Priority fees: 30%
        - TPS degradation: 20%
        - Slot skips: 10%
        """
        # Normalize components
        cu_score = min(cu_util, 1.0)
        
        # Priority fee score (log scale)
        fee_score = min(np.log10(priority_fee + 1) / 6, 1.0)  # 6 = log10(1M)
        
        # TPS score (lower is worse, assume 4000 is baseline)
        baseline_tps = 4000
        tps_score = max(0, 1 - (tps / baseline_tps))
        
        # Slot skip score
        skip_score = min(slot_skip * 10, 1.0)  # 10% skip rate = 1.0
        
        # Weighted combination
        stress = (
            0.40 * cu_score +
            0.30 * fee_score +
            0.20 * tps_score +
            0.10 * skip_score
        )
        
        return min(stress, 1.0)
    
    def _score_to_level(self, score: float) -> str:
        """Convert stress score to level."""
        if score < 0.25:
            return 'LOW'
        elif score < 0.50:
            return 'MODERATE'
        elif score < 0.75:
            return 'HIGH'
        elif score < 0.90:
            return 'CRITICAL'
        else:
            return 'CONGESTED'
    
    def _get_execution_recommendation(self) -> Dict[str, Any]:
        """
        Get execution recommendation based on stress level.
        """
        recommendations = {
            'LOW': {
                'mode': 'AGGRESSIVE',
                'size_multiplier': 1.0,
                'urgency': 'HIGH',
                'use_priority_fee': False,
            },
            'MODERATE': {
                'mode': 'NORMAL',
                'size_multiplier': 0.8,
                'urgency': 'MEDIUM',
                'use_priority_fee': False,
            },
            'HIGH': {
                'mode': 'PASSIVE',
                'size_multiplier': 0.5,
                'urgency': 'LOW',
                'use_priority_fee': True,
            },
            'CRITICAL': {
                'mode': 'HIDDEN',
                'size_multiplier': 0.25,
                'urgency': 'VERY_LOW',
                'use_priority_fee': True,
            },
            'CONGESTED': {
                'mode': 'ABORT',
                'size_multiplier': 0.0,
                'urgency': 'NONE',
                'use_priority_fee': False,
            },
        }
        
        return recommendations.get(self.current_stress_level, recommendations['LOW'])
    
    def get_trend(self) -> Dict[str, float]:
        """Get trend in network stress metrics."""
        if len(self.cu_history) < 10:
            return {'cu_trend': 0.0, 'fee_trend': 0.0, 'tps_trend': 0.0}
        
        x = np.arange(len(self.cu_history))
        
        cu_trend = np.polyfit(x, list(self.cu_history), 1)[0]
        fee_trend = np.polyfit(x, list(self.priority_fee_history), 1)[0]
        tps_trend = np.polyfit(x, list(self.tps_history), 1)[0]
        
        return {
            'cu_trend': cu_trend,
            'fee_trend': fee_trend,
            'tps_trend': tps_trend,
        }


# =============================================================================
# MAIN LOCAL COMPUTE ENGINE
# =============================================================================

class LocalComputeEngine:
    """
    LocalComputeEngine - The Edge Brain
    
    Integrates all local GPU-accelerated components:
    1. VPIN Calculator (toxicity detection)
    2. OFI Calculator (order flow imbalance)
    3. Local SLM (flash sentiment analysis)
    4. Solana Network Monitor (SOL-specific)
    
    Outputs unified microstructure metrics and escalation signals.
    """
    
    def __init__(
        self,
        vpin_bucket_size: float = 50000.0,
        vpin_n_buckets: int = 50,
        ofi_lookback: int = 100,
        slm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        escalation_sigma: float = 4.0,
        use_gpu: bool = True,
    ):
        # Check GPU availability
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            logger.info("RTX 5090 GPU acceleration ENABLED")
            # Set memory fraction
            try:
                cp.cuda.Device(0).use()
                mempool = cp.get_default_memory_pool()
                logger.info(f"GPU Memory: {mempool.used_bytes() / 1e9:.2f} GB used")
            except Exception as e:
                logger.warning(f"GPU setup warning: {e}")
        else:
            logger.warning("GPU acceleration DISABLED - using CPU fallback")
        
        # Initialize components
        self.vpin_calculators: Dict[str, GPUVPINCalculator] = {}
        self.ofi_calculator = GPUOFICalculator(
            lookback_periods=ofi_lookback,
            use_gpu=self.use_gpu,
        )
        self.sentiment_analyzer = LocalSLMSentimentAnalyzer(
            model_name=slm_model,
            escalation_sigma=escalation_sigma,
        )
        self.sol_network_monitor = SolanaNetworkStressMonitor()
        
        # Configuration
        self.vpin_bucket_size = vpin_bucket_size
        self.vpin_n_buckets = vpin_n_buckets
        
        # Assets to monitor
        self.assets = ['BTC', 'ETH', 'SOL']
        
        # Initialize VPIN for each asset
        for asset in self.assets:
            self.vpin_calculators[asset] = GPUVPINCalculator(
                bucket_size=vpin_bucket_size,
                n_buckets=vpin_n_buckets,
                use_gpu=self.use_gpu,
            )
        
        # Metrics cache
        self.latest_metrics: Dict[str, MicrostructureMetrics] = {}
        self.latest_sentiment: Optional[SentimentSignal] = None
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        
        logger.info("LocalComputeEngine initialized | Assets: " + ", ".join(self.assets))
    
    def _get_execution_mode(
        self,
        vpin: float,
        ofi_zscore: float,
        asset: str,
    ) -> str:
        """
        Determine recommended execution mode based on microstructure.
        
        Logic:
        - TOXIC VPIN (>0.7) -> HIDDEN/PASSIVE
        - High OFI against us -> PASSIVE
        - Low toxicity -> AGGRESSIVE
        """
        toxicity = self.vpin_calculators[asset].get_toxicity_level(vpin) if asset in self.vpin_calculators else 'LOW'
        
        # SOL-specific check
        if asset == 'SOL':
            sol_rec = self.sol_network_monitor._get_execution_recommendation()
            if sol_rec['mode'] == 'ABORT':
                return 'ABORT'
            if sol_rec['mode'] == 'HIDDEN':
                return 'HIDDEN'
        
        # VPIN-based decision
        if toxicity == 'TOXIC':
            return 'HIDDEN'
        elif toxicity == 'HIGH':
            return 'PASSIVE'
        elif toxicity == 'MODERATE':
            # Check OFI
            if abs(ofi_zscore) > 2.0:
                return 'PASSIVE'
            return 'AGGRESSIVE'
        else:
            return 'AGGRESSIVE'
    
    def process_trades(
        self,
        trades: List[Trade],
    ) -> Dict[str, float]:
        """
        Process batch of trades and update VPIN for each asset.
        
        Returns:
            Dict of asset -> current VPIN
        """
        start_time = time.time()
        results = {}
        
        # Group trades by asset
        trades_by_asset = {}
        for trade in trades:
            if trade.asset not in trades_by_asset:
                trades_by_asset[trade.asset] = []
            trades_by_asset[trade.asset].append(trade)
        
        # Process each asset
        for asset, asset_trades in trades_by_asset.items():
            if asset in self.vpin_calculators:
                vpin = self.vpin_calculators[asset].add_trades_batch(asset_trades)
                if vpin is not None:
                    results[asset] = vpin
        
        latency = (time.time() - start_time) * 1000
        self.processing_times.append(latency)
        
        return results
    
    def process_orderbook(
        self,
        book: OrderBookSnapshot,
    ) -> float:
        """
        Process orderbook update and return OFI.
        """
        return self.ofi_calculator.update(book)
    
    def process_sentiment(
        self,
        text: str,
        source: str = 'unknown',
    ) -> SentimentSignal:
        """
        Process text for sentiment analysis.
        
        Returns:
            SentimentSignal with escalation flag
        """
        signal = self.sentiment_analyzer.analyze(text, source)
        self.latest_sentiment = signal
        return signal
    
    def update_sol_network(
        self,
        cu_used: float,
        cu_limit: float,
        priority_fee: float,
        tps: float,
        slot_skip_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Update Solana network metrics.
        """
        return self.sol_network_monitor.update(
            cu_used, cu_limit, priority_fee, tps, slot_skip_rate
        )
    
    def get_microstructure_metrics(
        self,
        asset: str,
    ) -> MicrostructureMetrics:
        """
        Get current microstructure metrics for an asset.
        """
        # Get VPIN
        vpin = 0.5
        if asset in self.vpin_calculators:
            vpin = self.vpin_calculators[asset].calculate_vpin()
        
        # Get OFI
        ofi_stats = self.ofi_calculator.get_ofi_stats(asset)
        
        # Get toxicity level
        toxicity = 'LOW'
        if asset in self.vpin_calculators:
            toxicity = self.vpin_calculators[asset].get_toxicity_level(vpin)
        
        # Get execution mode
        exec_mode = self._get_execution_mode(vpin, ofi_stats['zscore'], asset)
        
        metrics = MicrostructureMetrics(
            timestamp=time.time(),
            asset=asset,
            vpin=vpin,
            ofi=ofi_stats['current'],
            ofi_cumulative=ofi_stats['cumulative'],
            toxicity_level=toxicity,
            recommended_mode=exec_mode,
        )
        
        self.latest_metrics[asset] = metrics
        return metrics
    
    def compute_full_update(
        self,
        trades: Optional[List[Trade]] = None,
        books: Optional[Dict[str, OrderBookSnapshot]] = None,
        texts: Optional[List[Tuple[str, str]]] = None,
        sol_network: Optional[Dict[str, float]] = None,
    ) -> LocalComputeResult:
        """
        Perform full compute update with all inputs.
        
        Args:
            trades: List of trades
            books: Dict of asset -> OrderBookSnapshot
            texts: List of (text, source) for sentiment
            sol_network: Dict with cu_used, cu_limit, priority_fee, tps
        
        Returns:
            Complete LocalComputeResult
        """
        start_time = time.time()
        
        # Process trades
        if trades:
            self.process_trades(trades)
        
        # Process orderbooks
        if books:
            for asset, book in books.items():
                self.process_orderbook(book)
        
        # Process sentiment
        sentiment = None
        escalation_required = False
        escalation_reason = None
        
        if texts:
            for text, source in texts:
                signal = self.process_sentiment(text, source)
                if signal.escalate:
                    escalation_required = True
                    escalation_reason = f"{signal.sigma_deviation:.1f}σ sentiment shock from {source}"
                    sentiment = signal
                    break  # First escalation wins
            
            if sentiment is None and texts:
                # Use last sentiment if no escalation
                sentiment = self.latest_sentiment
        
        # Update SOL network
        if sol_network:
            self.update_sol_network(**sol_network)
        
        # Get metrics for all assets
        microstructure = {}
        for asset in self.assets:
            microstructure[asset] = self.get_microstructure_metrics(asset)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LocalComputeResult(
            timestamp=time.time(),
            microstructure=microstructure,
            sentiment=sentiment,
            escalation_required=escalation_required,
            escalation_reason=escalation_reason,
            latency_ms=latency_ms,
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get processing performance statistics."""
        if len(self.processing_times) == 0:
            return {'mean_latency_ms': 0.0, 'p99_latency_ms': 0.0}
        
        times = list(self.processing_times)
        return {
            'mean_latency_ms': np.mean(times),
            'p50_latency_ms': np.percentile(times, 50),
            'p99_latency_ms': np.percentile(times, 99),
            'max_latency_ms': max(times),
            'samples': len(times),
        }
    
    def should_escalate_to_master(self) -> Tuple[bool, Optional[str]]:
        """
        Check if current state warrants escalation to Master Agent.
        
        Escalation triggers:
        1. Sentiment > 4σ
        2. VPIN > 0.8 (extreme toxicity)
        3. SOL network CONGESTED
        """
        reasons = []
        
        # Check sentiment
        if self.latest_sentiment and self.latest_sentiment.escalate:
            reasons.append(f"Sentiment {self.latest_sentiment.sigma_deviation:.1f}σ")
        
        # Check VPIN toxicity
        for asset, metrics in self.latest_metrics.items():
            if metrics.vpin > 0.8:
                reasons.append(f"{asset} VPIN {metrics.vpin:.2f}")
        
        # Check SOL network
        if self.sol_network_monitor.current_stress_level == 'CONGESTED':
            reasons.append("SOL network CONGESTED")
        
        if reasons:
            return True, " | ".join(reasons)
        return False, None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Demonstrate LocalComputeEngine usage."""
    
    # Initialize engine
    engine = LocalComputeEngine(
        vpin_bucket_size=50000.0,
        vpin_n_buckets=50,
        escalation_sigma=4.0,
        use_gpu=True,
    )
    
    # Simulate trades
    trades = [
        Trade(time.time() + i, 95000 + np.random.randn() * 100, 
              np.random.uniform(0.01, 1.0), TradeDirection.UNKNOWN, 'BTC')
        for i in range(100)
    ]
    
    # Simulate orderbook
    book = OrderBookSnapshot(
        timestamp=time.time(),
        asset='BTC',
        bids=np.array([[94990, 10], [94985, 20], [94980, 30]]),
        asks=np.array([[95010, 10], [95015, 20], [95020, 30]]),
    )
    
    # Simulate sentiment
    texts = [
        ("BTC breaking out! Moon soon! 🚀", "twitter"),
        ("Solana TPS hitting new highs", "news"),
    ]
    
    # Simulate SOL network
    sol_network = {
        'cu_used': 35e9,
        'cu_limit': 48e9,
        'priority_fee': 5000,
        'tps': 3500,
        'slot_skip_rate': 0.02,
    }
    
    # Full update
    result = engine.compute_full_update(
        trades=trades,
        books={'BTC': book},
        texts=texts,
        sol_network=sol_network,
    )
    
    print("\n=== LocalComputeEngine Results ===")
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Escalation Required: {result.escalation_required}")
    if result.escalation_reason:
        print(f"Escalation Reason: {result.escalation_reason}")
    
    print("\nMicrostructure Metrics:")
    for asset, metrics in result.microstructure.items():
        print(f"  {asset}: VPIN={metrics.vpin:.3f}, "
              f"Toxicity={metrics.toxicity_level}, "
              f"Mode={metrics.recommended_mode}")
    
    if result.sentiment:
        print(f"\nSentiment: {result.sentiment.sentiment_score:.2f} "
              f"({result.sentiment.sigma_deviation:.1f}σ)")
    
    print("\nPerformance Stats:")
    stats = engine.get_performance_stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}")


if __name__ == '__main__':
    example_usage()

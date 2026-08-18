"""
SOTA v3.1 Lead-Lag Alpha Engine
================================

Multi-exchange lead-lag signal extraction:
- Binance Perpetuals: Taker Volume (lead indicator)
- Deribit Options: DVOL implied volatility
- Kraken: Execution venue (lags by 50-200ms)

Features:
- 200ms Half-life audit for signal decay
- 3σ taker volume spike detection
- Front-running execution protocols
- Real-time latency measurement
"""

import time
import asyncio
import logging
import numpy as np
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import json

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LeadLagEngine')


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = auto()
    DERIBIT = auto()
    KRAKEN = auto()


# WebSocket endpoints
WS_ENDPOINTS = {
    Exchange.BINANCE: "wss://fstream.binance.com/ws",
    Exchange.DERIBIT: "wss://www.deribit.com/ws/api/v2",
    Exchange.KRAKEN: "wss://ws.kraken.com",
}

# Signal half-life threshold (milliseconds)
SIGNAL_HALF_LIFE_MS = 200.0

# Taker volume spike threshold (standard deviations)
TAKER_SPIKE_THRESHOLD_SIGMA = 3.0

# Lookback for statistics
LOOKBACK_TICKS = 100


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class TakerTrade(NamedTuple):
    """Individual taker trade."""
    timestamp_ms: float
    price: float
    quantity: float
    is_buyer_maker: bool  # True = sell aggressor, False = buy aggressor


@dataclass
class TakerVolumeState:
    """State for taker volume tracking."""
    
    # Recent trades
    trades: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Rolling statistics (1-minute windows)
    buy_volume_1m: float = 0.0
    sell_volume_1m: float = 0.0
    net_taker_volume_1m: float = 0.0
    
    # Historical statistics for z-score
    volume_history: deque = field(default_factory=lambda: deque(maxlen=LOOKBACK_TICKS))
    
    # Spike detection
    volume_mean: float = 0.0
    volume_std: float = 1.0
    current_zscore: float = 0.0
    is_spike: bool = False
    
    # Timing
    last_update_ms: float = 0.0
    latency_to_kraken_ms: float = 0.0


@dataclass
class DVOLState:
    """State for Deribit DVOL tracking."""
    
    btc_dvol: float = 0.0
    eth_dvol: float = 0.0
    
    btc_dvol_change_1h: float = 0.0
    eth_dvol_change_1h: float = 0.0
    
    # Historical
    btc_dvol_history: deque = field(default_factory=lambda: deque(maxlen=60))
    eth_dvol_history: deque = field(default_factory=lambda: deque(maxlen=60))
    
    last_update_ms: float = 0.0


@dataclass
class LeadLagSignal:
    """Output signal from lead-lag analysis."""
    
    asset: str
    
    # Signal strength [-1, +1]
    signal: float = 0.0
    
    # Confidence [0, 1]
    confidence: float = 0.0
    
    # Components
    taker_volume_zscore: float = 0.0
    dvol_change: float = 0.0
    price_divergence_bps: float = 0.0
    
    # Timing
    lead_latency_ms: float = 0.0
    half_life_valid: bool = True
    
    # Execution recommendation
    front_run_recommended: bool = False
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH, CRITICAL
    
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BINANCE TAKER VOLUME MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class BinanceTakerMonitor:
    """
    Monitor Binance Perpetuals for taker volume signals.
    
    Subscriptions:
    - @aggTrade: Aggregated trades (taker direction)
    - @markPrice: Mark price for divergence detection
    """
    
    def __init__(self, assets: List[str] = ['BTC', 'ETH', 'SOL']):
        self.assets = assets
        self.logger = logging.getLogger('BinanceTaker')
        
        # State per asset
        self.states: Dict[str, TakerVolumeState] = {
            asset: TakerVolumeState() for asset in assets
        }
        
        # Prices
        self.binance_prices: Dict[str, float] = {asset: 0.0 for asset in assets}
        self.kraken_prices: Dict[str, float] = {asset: 0.0 for asset in assets}
        
        # WebSocket
        self.ws = None
        self._running = False
        # [P303] _reconnect_ws reads _reconnect_count, and it is no longer
        # reachable only from process_messages (which used to initialise it).
        self._reconnect_count = 0
        self._connect_failures = 0

    def _get_symbol(self, asset: str) -> str:
        """Convert asset to Binance symbol."""
        return f"{asset.lower()}usdt"
    
    async def connect(self):
        """Connect to Binance WebSocket."""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("websockets library not available")
            return
        
        # Build subscription streams
        streams = []
        for asset in self.assets:
            symbol = self._get_symbol(asset)
            streams.append(f"{symbol}@aggTrade")
            streams.append(f"{symbol}@markPrice@1s")
        
        url = f"{WS_ENDPOINTS[Exchange.BINANCE]}/{'/'.join(streams)}"
        
        self.logger.info(f"Connecting to Binance: {url[:80]}...")
        
        # [P303] Arm the message loop BEFORE the attempt. `_running` used to be
        # set only on the SUCCESS path, and process_messages gates on
        # `while self._running:` — so a failed INITIAL connect left the monitor
        # permanently dead: the loop returned immediately, `_reconnect_ws` was
        # never reached, and the L4-03 "auto-reconnect" could not fire in the
        # one case it was written for (a venue outage at boot). Nothing above
        # recovers it either — main.py's restart guard reads the ENGINE's
        # `_running`, which `start()` sets unconditionally, so the engine
        # reports healthy while both its monitors are dead. Setting it first
        # costs nothing: close()/stop() still set it False to stop the loop.
        self._running = True
        try:
            self.ws = await websockets.connect(url)
            self.logger.info("Connected to Binance WebSocket")
            self._connect_failures = 0
        except Exception as e:
            # [P303] WARNING, not ERROR. ERROR is forwarded to Discord, and a
            # venue-side outage is something the operator can neither act on
            # nor be harmed by while the reconnect loop is running — the
            # P202/P240 shape (an alert whose only resolutions are theatre or
            # ignoring it, which is how a real CRITICAL stops being read). A
            # SUSTAINED failure is a different claim and still escalates.
            self._connect_failures = getattr(self, "_connect_failures", 0) + 1
            _msg = (f"[LEAD_LAG] Binance connect failed "
                    f"(attempt {self._connect_failures}): "
                    f"{type(e).__name__}: {e} — reconnect loop is armed; "
                    f"until it succeeds the taker-flow inputs go stale")
            if self._connect_failures >= 4:
                self.logger.error(_msg + " — SUSTAINED, no longer a blip")
            else:
                self.logger.warning(_msg)

    async def process_messages(self):
        """Process incoming WebSocket messages with auto-reconnect (L4-03)."""
        self._reconnect_count = 0
        while self._running:
            if not self.ws:
                await self._reconnect_ws()
                continue
            try:
                async for message in self.ws:
                    if not self._running:
                        return
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                        self._reconnect_count = 0
                    except Exception as e:
                        self.logger.error(f"Message processing error: {e}")
            except Exception as e:
                if not self._running:
                    return
                self.logger.warning(f"[LEAD_LAG] Binance WS disconnected: {e}")
                self.ws = None
                await self._reconnect_ws()

    async def _reconnect_ws(self):
        """Reconnect Binance WS with exponential backoff."""
        delay = min(30, 2 ** self._reconnect_count)
        self._reconnect_count += 1
        self.logger.info(f"[LEAD_LAG] Binance reconnect in {delay}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"[LEAD_LAG] Binance reconnect failed: {e}")
            self.ws = None

    async def _handle_message(self, data: dict):
        """Handle a single WebSocket message."""
        event_type = data.get('e')
        
        if event_type == 'aggTrade':
            await self._handle_agg_trade(data)
        elif event_type == 'markPriceUpdate':
            await self._handle_mark_price(data)
    
    async def _handle_agg_trade(self, data: dict):
        """Handle aggregated trade message."""
        symbol = data.get('s', '').upper()
        asset = symbol.replace('USDT', '')
        
        if asset not in self.states:
            return
        
        state = self.states[asset]
        
        # Parse trade
        trade = TakerTrade(
            timestamp_ms=data['T'],
            price=float(data['p']),
            quantity=float(data['q']),
            is_buyer_maker=data['m']
        )
        
        state.trades.append(trade)
        state.last_update_ms = time.time() * 1000
        
        # Update rolling volumes
        self._update_rolling_volumes(state)
        
        # Calculate z-score
        self._update_zscore(state)
    
    async def _handle_mark_price(self, data: dict):
        """Handle mark price update."""
        symbol = data.get('s', '').upper()
        asset = symbol.replace('USDT', '')
        
        if asset in self.binance_prices:
            self.binance_prices[asset] = float(data.get('p', 0))
    
    def _update_rolling_volumes(self, state: TakerVolumeState):
        """Update 1-minute rolling volumes."""
        now_ms = time.time() * 1000
        cutoff_ms = now_ms - 60000  # 1 minute
        
        buy_vol = 0.0
        sell_vol = 0.0
        
        for trade in state.trades:
            if trade.timestamp_ms >= cutoff_ms:
                if trade.is_buyer_maker:
                    sell_vol += trade.quantity * trade.price
                else:
                    buy_vol += trade.quantity * trade.price
        
        state.buy_volume_1m = buy_vol
        state.sell_volume_1m = sell_vol
        state.net_taker_volume_1m = buy_vol - sell_vol
    
    def _update_zscore(self, state: TakerVolumeState):
        """Update taker volume z-score."""
        # Add to history
        total_volume = state.buy_volume_1m + state.sell_volume_1m
        state.volume_history.append(total_volume)
        
        if len(state.volume_history) < 10:
            return
        
        # Calculate statistics
        volumes = np.array(state.volume_history)
        state.volume_mean = np.mean(volumes)
        state.volume_std = np.std(volumes) + 1e-10  # Avoid division by zero
        
        # Current z-score
        state.current_zscore = (total_volume - state.volume_mean) / state.volume_std
        
        # Spike detection
        state.is_spike = abs(state.current_zscore) > TAKER_SPIKE_THRESHOLD_SIGMA
    
    def update_kraken_price(self, asset: str, price: float):
        """Update Kraken price for latency calculation."""
        self.kraken_prices[asset] = price
    
    def get_state(self, asset: str) -> Optional[TakerVolumeState]:
        """Get current state for an asset."""
        return self.states.get(asset)
    
    def get_price_divergence_bps(self, asset: str) -> float:
        """Calculate price divergence between Binance and Kraken in bps."""
        binance = self.binance_prices.get(asset, 0)
        kraken = self.kraken_prices.get(asset, 0)
        
        if binance == 0 or kraken == 0:
            return 0.0
        
        return ((binance - kraken) / kraken) * 10000  # bps
    
    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        if self.ws:
            await self.ws.close()


# ═══════════════════════════════════════════════════════════════════════════════
# DERIBIT DVOL MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class DeribitDVOLMonitor:
    """
    Monitor Deribit for DVOL (implied volatility index).
    
    DVOL is a forward-looking volatility measure that often leads spot moves.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('DeribitDVOL')
        self.state = DVOLState()
        self.ws = None
        self._running = False
        # [P303] _reconnect_ws reads _reconnect_count, and it is no longer
        # reachable only from process_messages (which used to initialise it).
        self._reconnect_count = 0
        self._connect_failures = 0
    
    async def connect(self):
        """Connect to Deribit WebSocket."""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.error("websockets library not available")
            return
        
        url = WS_ENDPOINTS[Exchange.DERIBIT]
        
        self.logger.info(f"Connecting to Deribit: {url}")
        
        # [P303] Same defect and same fix as BinanceTakerMonitor.connect — this
        # is the monitor that exposed it. Deribit returned HTTP 503 during the
        # 2026-08-18 10:03 boot; measured as a VENDOR-side outage (503 from the
        # server AND from an unrelated host, on REST and WS alike, so not a
        # block on us and not the P293b Cloudflare client-signature trap), and
        # DVOL would have stayed dead for the whole process lifetime even after
        # Deribit recovered.
        self._running = True
        try:
            self.ws = await websockets.connect(url)

            # Subscribe to DVOL indices
            await self._subscribe()

            self.logger.info("Connected to Deribit WebSocket")
            self._connect_failures = 0
        except Exception as e:
            # [P303] WARNING, not ERROR — see the Binance sibling. DVOL has no
            # live consumer today (`dvol_to_market_data` is deliberately absent
            # per P298: Deribit publishes an INDEX LEVEL, and the constitution
            # reads that field as a z-score, so arming it would fire
            # EXTREME_DVOL every tick). Its only effect is the lead-lag
            # confidence modifier, which is exactly 1.0 when DVOL is 0.0.
            # Threshold is higher than Binance's: DVOL is advisory, taker flow
            # is not.
            self._connect_failures = getattr(self, "_connect_failures", 0) + 1
            _msg = (f"[LEAD_LAG] Deribit connect failed "
                    f"(attempt {self._connect_failures}): "
                    f"{type(e).__name__}: {e} — reconnect loop is armed; until "
                    f"it succeeds DVOL stays 0.0, which leaves the lead-lag "
                    f"confidence modifier at a neutral 1.0 (no live consumer)")
            if self._connect_failures >= 8:
                self.logger.error(_msg + " — SUSTAINED, no longer a blip")
            else:
                self.logger.warning(_msg)
    
    async def _subscribe(self):
        """Subscribe to DVOL channels."""
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {
                "channels": [
                    "deribit_volatility_index.btc_usd",
                    "deribit_volatility_index.eth_usd"
                ]
            }
        }
        
        await self.ws.send(json.dumps(subscribe_msg))

    async def process_messages(self):
        """Process incoming WebSocket messages with auto-reconnect (L4-03)."""
        self._reconnect_count = 0
        while self._running:
            if not self.ws:
                await self._reconnect_ws()
                continue
            try:
                async for message in self.ws:
                    if not self._running:
                        return
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                        self._reconnect_count = 0
                    except Exception as e:
                        self.logger.error(f"Message processing error: {e}")
            except Exception as e:
                if not self._running:
                    return
                self.logger.warning(f"[LEAD_LAG] Deribit WS disconnected: {e}")
                self.ws = None
                await self._reconnect_ws()

    async def _reconnect_ws(self):
        """Reconnect Deribit WS with exponential backoff."""
        delay = min(30, 2 ** self._reconnect_count)
        self._reconnect_count += 1
        self.logger.info(f"[LEAD_LAG] Deribit reconnect in {delay}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)
        try:
            await self.connect()
        except Exception as e:
            self.logger.error(f"[LEAD_LAG] Deribit reconnect failed: {e}")
            self.ws = None

    async def _handle_message(self, data: dict):
        """Handle a single WebSocket message."""
        if 'params' not in data:
            return
        
        params = data['params']
        channel = params.get('channel', '')
        
        if 'deribit_volatility_index' in channel:
            dvol_data = params.get('data', {})
            
            if 'btc' in channel.lower():
                old_dvol = self.state.btc_dvol
                self.state.btc_dvol = dvol_data.get('volatility', 0)
                self.state.btc_dvol_history.append(self.state.btc_dvol)
                
                if len(self.state.btc_dvol_history) >= 60:
                    self.state.btc_dvol_change_1h = (
                        self.state.btc_dvol - self.state.btc_dvol_history[0]
                    )
            
            elif 'eth' in channel.lower():
                old_dvol = self.state.eth_dvol
                self.state.eth_dvol = dvol_data.get('volatility', 0)
                self.state.eth_dvol_history.append(self.state.eth_dvol)
                
                if len(self.state.eth_dvol_history) >= 60:
                    self.state.eth_dvol_change_1h = (
                        self.state.eth_dvol - self.state.eth_dvol_history[0]
                    )
            
            self.state.last_update_ms = time.time() * 1000
    
    def get_state(self) -> DVOLState:
        """Get current DVOL state."""
        return self.state
    
    async def close(self):
        """Close WebSocket connection."""
        self._running = False
        if self.ws:
            await self.ws.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LEAD-LAG ALPHA ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LeadLagAlphaEngine:
    """
    Lead-Lag Alpha Engine for cross-exchange signal extraction.
    
    Core Logic:
    1. Monitor Binance taker volume for >3σ spikes
    2. Track Deribit DVOL for volatility regime changes
    3. Calculate price divergence between Binance and Kraken
    4. Apply 200ms half-life decay to invalidate stale signals
    5. Generate front-running execution recommendations
    """
    
    def __init__(self, assets: List[str] = ['BTC', 'ETH', 'SOL']):
        self.assets = assets
        self.logger = logging.getLogger('LeadLagEngine')
        
        # Monitors
        self.binance_monitor = BinanceTakerMonitor(assets)
        self.deribit_monitor = DeribitDVOLMonitor()
        
        # Timing
        self.signal_timestamps: Dict[str, float] = {asset: 0.0 for asset in assets}
        self.kraken_update_timestamps: Dict[str, float] = {asset: 0.0 for asset in assets}
        
        # State
        self._running = False
    
    async def start(self):
        """Start all monitors."""
        self.logger.info("Starting Lead-Lag Alpha Engine...")
        self._running = True

        # Connect to exchanges
        await asyncio.gather(
            self.binance_monitor.connect(),
            self.deribit_monitor.connect()
        )

        # [P37 2026-04-24] Was discarded asyncio.create_task() result → tasks
        # could be GC'd mid-WebSocket loop, AND stop() had no way to cancel
        # them. Now stored as instance attrs so stop() can cancel cleanly.
        self._binance_task = asyncio.create_task(
            self.binance_monitor.process_messages(),
            name="lead_lag_binance_messages",
        )
        self._deribit_task = asyncio.create_task(
            self.deribit_monitor.process_messages(),
            name="lead_lag_deribit_messages",
        )

        self.logger.info("Lead-Lag Alpha Engine started")

    async def stop(self):
        """Stop all monitors."""
        self._running = False

        # [P37 2026-04-24] Cancel processing tasks before closing connections
        # so the message loops exit cleanly instead of dangling.
        for _attr in ("_binance_task", "_deribit_task"):
            _t = getattr(self, _attr, None)
            if _t and not _t.done():
                _t.cancel()
                try:
                    await _t
                except (asyncio.CancelledError, Exception):
                    pass

        await asyncio.gather(
            self.binance_monitor.close(),
            self.deribit_monitor.close()
        )

        self.logger.info("Lead-Lag Alpha Engine stopped")
    
    def update_kraken_price(self, asset: str, price: float, timestamp_ms: float):
        """Update Kraken price for latency calculation."""
        self.binance_monitor.update_kraken_price(asset, price)
        self.kraken_update_timestamps[asset] = timestamp_ms
    
    def _calculate_half_life_decay(self, signal_age_ms: float) -> float:
        """
        Calculate signal decay factor based on half-life.
        
        Decay = 2^(-age / half_life)
        At 200ms: decay = 0.5
        At 400ms: decay = 0.25
        At 600ms: decay = 0.125
        """
        if signal_age_ms <= 0:
            return 1.0
        
        return 2.0 ** (-signal_age_ms / SIGNAL_HALF_LIFE_MS)
    
    def _is_half_life_valid(self, signal_age_ms: float) -> bool:
        """Check if signal is within valid half-life window."""
        return signal_age_ms < SIGNAL_HALF_LIFE_MS
    
    def generate_signal(self, asset: str) -> LeadLagSignal:
        """
        Generate lead-lag signal for an asset.
        
        Signal Composition:
        1. Taker volume z-score: Buy pressure -> positive, Sell pressure -> negative
        2. DVOL change: Rising vol -> uncertainty, Falling vol -> conviction
        3. Price divergence: Binance leads -> front-running opportunity
        
        Half-Life Audit:
        - If signal age > 200ms, invalidate and reduce confidence
        """
        signal = LeadLagSignal(asset=asset, timestamp=time.time())
        
        # Get states
        taker_state = self.binance_monitor.get_state(asset)
        dvol_state = self.deribit_monitor.get_state()
        
        if not taker_state:
            return signal
        
        # Calculate signal age (latency from Binance to now)
        now_ms = time.time() * 1000
        signal_age_ms = now_ms - taker_state.last_update_ms
        
        # Half-life validation
        signal.lead_latency_ms = signal_age_ms
        signal.half_life_valid = self._is_half_life_valid(signal_age_ms)
        
        # Decay factor
        decay = self._calculate_half_life_decay(signal_age_ms)
        
        # === Component 1: Taker Volume ===
        signal.taker_volume_zscore = taker_state.current_zscore
        
        # Positive z-score = more buying = bullish
        # Negative z-score = more selling = bearish
        taker_signal = np.tanh(taker_state.current_zscore / 3.0)  # Normalize to [-1, +1]
        
        # === Component 2: DVOL Change ===
        if asset == 'BTC':
            dvol_change = dvol_state.btc_dvol_change_1h
        elif asset == 'ETH':
            dvol_change = dvol_state.eth_dvol_change_1h
        else:
            dvol_change = 0.0  # No SOL DVOL
        
        signal.dvol_change = dvol_change
        
        # Rising DVOL -> reduce confidence, Falling DVOL -> increase confidence
        dvol_confidence_modifier = 1.0 - np.tanh(dvol_change / 10.0) * 0.3
        
        # === Component 3: Price Divergence ===
        price_div_bps = self.binance_monitor.get_price_divergence_bps(asset)
        signal.price_divergence_bps = price_div_bps
        
        # Binance > Kraken -> front-run opportunity (buy before Kraken catches up)
        # Binance < Kraken -> Kraken is stale or front-running sell
        divergence_signal = np.tanh(price_div_bps / 10.0)  # 10 bps = ~0.46
        
        # === Combine Signals ===
        # Weight: Taker (60%), Divergence (30%), DVOL (10%)
        raw_signal = (
            0.60 * taker_signal +
            0.30 * divergence_signal +
            0.10 * (-np.sign(dvol_change) * min(abs(dvol_change) / 20.0, 1.0))
        )
        
        # Apply half-life decay
        signal.signal = raw_signal * decay
        
        # === Confidence ===
        # Base confidence from taker volume spike
        base_confidence = min(abs(taker_state.current_zscore) / TAKER_SPIKE_THRESHOLD_SIGMA, 1.0)
        
        # Reduce confidence if half-life exceeded
        if not signal.half_life_valid:
            base_confidence *= 0.5
        
        # Reduce confidence if DVOL is rising
        base_confidence *= dvol_confidence_modifier
        
        signal.confidence = base_confidence
        
        # === Execution Recommendation ===
        signal.front_run_recommended = (
            taker_state.is_spike and
            signal.half_life_valid and
            abs(price_div_bps) > 5.0  # At least 5 bps divergence
        )
        
        # Urgency
        if taker_state.is_spike and signal.half_life_valid and abs(price_div_bps) > 10.0:
            signal.urgency = "CRITICAL"
        elif taker_state.is_spike and signal.half_life_valid:
            signal.urgency = "HIGH"
        elif abs(taker_state.current_zscore) > 2.0:
            signal.urgency = "NORMAL"
        else:
            signal.urgency = "LOW"
        
        return signal
    
    def generate_all_signals(self) -> Dict[str, LeadLagSignal]:
        """Generate signals for all assets."""
        return {asset: self.generate_signal(asset) for asset in self.assets}


# ═══════════════════════════════════════════════════════════════════════════════
# FRONT-RUNNING EXECUTION PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrontRunDecision:
    """Decision for front-running execution."""
    
    asset: str
    action: str  # "BUY", "SELL", "HOLD"
    urgency: str  # "LOW", "NORMAL", "HIGH", "CRITICAL"
    
    # Sizing
    size_multiplier: float = 1.0  # 0.5x to 2.0x normal size
    
    # Execution
    use_taker: bool = False  # True = aggressive taker, False = passive maker
    price_improvement_bps: float = 0.0  # How far to improve price
    
    # Timing
    max_execution_time_ms: float = 500.0  # Maximum time to fill
    
    # Reason
    reason: str = ""


class FrontRunningProtocol:
    """
    Protocol for front-running execution based on lead-lag signals.
    
    Rules:
    1. CRITICAL urgency -> Aggressive taker, 2x size, immediate
    2. HIGH urgency -> Taker if VPIN low, else passive at improved price
    3. NORMAL urgency -> Passive at best bid/ask
    4. LOW urgency -> No action
    
    Safety:
    - Never front-run if VPIN is toxic
    - Never front-run if signal is stale (>200ms)
    - Size caps based on liquidity
    """
    
    def __init__(self, max_size_usd: float = 50000.0):
        self.max_size_usd = max_size_usd
        self.logger = logging.getLogger('FrontRunProtocol')
    
    def decide(
        self,
        signal: LeadLagSignal,
        vpin: float = 0.0,
        vpin_toxic_threshold: float = 0.7
    ) -> FrontRunDecision:
        """Generate front-running execution decision."""
        
        decision = FrontRunDecision(
            asset=signal.asset,
            action="HOLD",
            urgency=signal.urgency
        )
        
        # Safety checks
        if not signal.half_life_valid:
            decision.reason = "Signal stale (>200ms)"
            return decision
        
        if vpin > vpin_toxic_threshold:
            decision.reason = f"VPIN toxic ({vpin:.2f})"
            return decision
        
        if signal.confidence < 0.3:
            decision.reason = f"Low confidence ({signal.confidence:.2f})"
            return decision
        
        # Determine action
        if signal.signal > 0.2:
            decision.action = "BUY"
        elif signal.signal < -0.2:
            decision.action = "SELL"
        else:
            decision.reason = "Signal too weak"
            return decision
        
        # Sizing based on urgency and confidence
        if signal.urgency == "CRITICAL":
            decision.size_multiplier = min(2.0, 1.0 + signal.confidence)
            decision.use_taker = True
            decision.max_execution_time_ms = 100.0
            decision.reason = "Critical urgency - aggressive taker"
        
        elif signal.urgency == "HIGH":
            decision.size_multiplier = min(1.5, 1.0 + signal.confidence * 0.5)
            decision.use_taker = vpin < 0.3  # Taker only if very safe
            decision.price_improvement_bps = 2.0 if not decision.use_taker else 0.0
            decision.max_execution_time_ms = 200.0
            decision.reason = "High urgency - conditional taker"
        
        elif signal.urgency == "NORMAL":
            decision.size_multiplier = 1.0
            decision.use_taker = False
            decision.price_improvement_bps = 1.0
            decision.max_execution_time_ms = 500.0
            decision.reason = "Normal urgency - passive maker"
        
        else:  # LOW
            decision.size_multiplier = 0.5
            decision.use_taker = False
            decision.price_improvement_bps = 0.5
            decision.max_execution_time_ms = 1000.0
            decision.reason = "Low urgency - reduced size passive"
        
        return decision


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_lead_lag_engine(assets: List[str] = None) -> LeadLagAlphaEngine:
    """Factory function for LeadLagAlphaEngine."""
    if assets is None:
        assets = ['BTC', 'ETH', 'SOL']
    return LeadLagAlphaEngine(assets)


def create_front_running_protocol(max_size_usd: float = 50000.0) -> FrontRunningProtocol:
    """Factory function for FrontRunningProtocol."""
    return FrontRunningProtocol(max_size_usd)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("SOTA v3.1 Lead-Lag Alpha Engine")
    print("="*60)
    print()
    
    # Test signal generation (simulated)
    print("Testing signal generation (simulated)...")
    
    # Create engine
    engine = create_lead_lag_engine()
    
    # Simulate taker volume spike
    btc_state = engine.binance_monitor.states['BTC']
    btc_state.current_zscore = 4.5  # >3σ spike
    btc_state.is_spike = True
    btc_state.last_update_ms = time.time() * 1000 - 50  # 50ms ago
    
    # Simulate prices
    engine.binance_monitor.binance_prices['BTC'] = 97000.0
    engine.binance_monitor.kraken_prices['BTC'] = 96980.0  # 20 bps divergence
    
    # Generate signal
    signal = engine.generate_signal('BTC')
    
    print(f"BTC Signal:")
    print(f"  Signal: {signal.signal:.4f}")
    print(f"  Confidence: {signal.confidence:.4f}")
    print(f"  Taker Z-Score: {signal.taker_volume_zscore:.2f}")
    print(f"  Price Divergence: {signal.price_divergence_bps:.2f} bps")
    print(f"  Latency: {signal.lead_latency_ms:.1f} ms")
    print(f"  Half-Life Valid: {signal.half_life_valid}")
    print(f"  Front-Run Recommended: {signal.front_run_recommended}")
    print(f"  Urgency: {signal.urgency}")
    print()
    
    # Test front-running protocol
    print("Testing front-running protocol...")
    protocol = create_front_running_protocol()
    decision = protocol.decide(signal, vpin=0.2)
    
    print(f"Decision:")
    print(f"  Action: {decision.action}")
    print(f"  Size Multiplier: {decision.size_multiplier:.2f}x")
    print(f"  Use Taker: {decision.use_taker}")
    print(f"  Max Execution Time: {decision.max_execution_time_ms:.0f} ms")
    print(f"  Reason: {decision.reason}")
    print()
    
    # Test half-life decay
    print("Testing half-life decay...")
    for age_ms in [0, 100, 200, 300, 400, 500]:
        decay = engine._calculate_half_life_decay(age_ms)
        valid = engine._is_half_life_valid(age_ms)
        print(f"  Age {age_ms:3d}ms: decay={decay:.4f}, valid={valid}")
    print()
    
    print("All tests passed!")

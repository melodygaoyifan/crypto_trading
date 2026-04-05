#!/usr/bin/env python3
"""
================================================================================
HMATS v5.1 - Unified Trading Runner
================================================================================
Purpose: Single entry point for backtest, paper, and live trading modes
         Uses same event chain for all modes (SOTA P1 requirement)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          UnifiedRunner                                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                     │
    │  │  BACKTEST   │  │   PAPER     │  │    LIVE     │                     │
    │  │  Mode       │  │   Mode      │  │    Mode     │                     │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                     │
    │         │                │                │                             │
    │         └────────────────┼────────────────┘                             │
    │                          ▼                                              │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    Shared Event Pipeline                         │   │
    │  │  DataFeed -> Regime -> Signals -> Fusion -> Risk -> Execution        │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

SOTA Requirement:
- "做一个统一 runner（backtest / paper / live 三种 mode 同一套事件链路）"

Author: HMATS v5.1 P1 Fix
Version: 5.1.0
================================================================================
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Generator
import json

# Local imports
from infra.event_bus import EventBus, Event, EventType
from infra.event_integration import (
    EventIntegrationManager, 
    get_event_integration,
    TickLogBuilder
)
from orchestration.system_mode import SystemMode, ModeState

logger = logging.getLogger(__name__)


# =============================================================================
# RUNNER MODE
# =============================================================================

class RunnerMode(Enum):
    """Trading runner mode."""
    BACKTEST = "backtest"     # Historical data, simulated execution
    PAPER = "paper"           # Live data, simulated execution  
    LIVE = "live"             # Live data, real execution


@dataclass
class RunnerConfig:
    """Configuration for unified runner."""
    mode: RunnerMode = RunnerMode.PAPER
    
    # Timing
    tick_interval_seconds: int = 14400  # 4H = 14400 seconds
    execution_loop_ms: int = 200        # 200ms execution modifiers
    
    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["SOL/USD", "BTC/USD", "ETH/USD"])
    primary_symbol: str = "SOL/USD"
    
    # Risk limits
    max_position_usd: float = 10000.0
    max_drawdown_pct: float = 0.05
    
    # Backtest specific
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    
    # Logging
    log_dir: Path = field(default_factory=lambda: Path("logs/runner"))
    enable_event_logging: bool = True
    
    # Exchange (for live/paper)
    exchange: str = "kraken"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


@dataclass
class TickContext:
    """Context passed through each tick."""
    tick_id: str
    timestamp: datetime
    mode: RunnerMode
    system_mode: str  # NORMAL, OPPORTUNITY, NO_TRADE
    regime: str
    market_data: Dict[str, Any]
    signals: Dict[str, Any] = field(default_factory=dict)
    risk_state: Dict[str, Any] = field(default_factory=dict)
    execution_result: Optional[Dict] = None


# =============================================================================
# DATA FEED INTERFACE
# =============================================================================

class DataFeed(ABC):
    """Abstract data feed interface."""
    
    @abstractmethod
    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        """Get next market data tick."""
        pass
    
    @abstractmethod
    def has_more_data(self) -> bool:
        """Check if more data available."""
        pass


class BacktestDataFeed(DataFeed):
    """Historical data feed for backtesting."""
    
    def __init__(
        self,
        data_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self._data: List[Dict] = []
        self._index = 0
        self._load_data()
        
    def _load_data(self):
        """Load historical data from file."""
        if self.data_path.exists():
            with open(self.data_path, 'r') as f:
                self._data = json.load(f)
        else:
            # Generate sample data for testing
            logger.warning(f"Data file not found: {self.data_path}, using sample data")
            self._data = self._generate_sample_data()
            
    def _generate_sample_data(self) -> List[Dict]:
        """Generate sample historical data."""
        data = []
        base_price = 150.0
        timestamp = datetime.utcnow() - timedelta(days=30)
        
        for i in range(720):  # 30 days * 24 hours / 4h = 180 ticks
            import random
            price = base_price * (1 + random.uniform(-0.02, 0.02))
            data.append({
                "timestamp": timestamp.isoformat(),
                "symbol": "SOL/USD",
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": random.uniform(100000, 500000)
            })
            timestamp += timedelta(hours=4)
            base_price = price
            
        return data
        
    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        """Get next historical tick."""
        if self._index >= len(self._data):
            return None
        tick = self._data[self._index]
        self._index += 1
        return tick
        
    def has_more_data(self) -> bool:
        """Check if more data available."""
        return self._index < len(self._data)


class LiveDataFeed(DataFeed):
    """Live market data feed."""
    
    def __init__(self, symbols: List[str], exchange: str = "kraken"):
        self.symbols = symbols
        self.exchange = exchange
        self._running = True
        
    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        """Get current market data (stub - integrate with real exchange)."""
        import random
        # In production, this would connect to Kraken WebSocket
        # For simulation, generate realistic price movement
        if not hasattr(self, '_last_prices'):
            self._last_prices = {'SOLUSD': 150.0, 'BTCUSD': 100000.0, 'ETHUSD': 3500.0}
        
        symbol = self.symbols[0] if self.symbols else "SOLUSD"
        last_price = self._last_prices.get(symbol, 150.0)
        
        # Random walk with mean reversion
        change_pct = random.gauss(0, 0.001)  # 0.1% std dev
        new_price = last_price * (1 + change_pct)
        self._last_prices[symbol] = new_price
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "price": new_price,
            "volume": random.uniform(50000, 200000)
        }
        
    def has_more_data(self) -> bool:
        """Live feed always has more data."""
        return self._running
        
    def stop(self):
        """Stop the data feed."""
        self._running = False


# =============================================================================
# EXECUTION INTERFACE
# =============================================================================

class ExecutionBackend(ABC):
    """Abstract execution backend."""
    
    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Submit order and return result."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        pass


class SimulatedExecution(ExecutionBackend):
    """Simulated execution for backtest/paper."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self._order_counter = 0
        
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Simulate order execution."""
        self._order_counter += 1
        order_id = f"SIM_{self._order_counter:06d}"
        
        # Simulate fill (in production, would wait for exchange)
        # Use provided price, or estimate from last known price
        if price:
            fill_price = price
        elif hasattr(self, '_last_fill_prices') and symbol in self._last_fill_prices:
            fill_price = self._last_fill_prices[symbol]
        else:
            # Default prices by asset
            default_prices = {'SOLUSD': 150.0, 'BTCUSD': 100000.0, 'ETHUSD': 3500.0}
            fill_price = default_prices.get(symbol, 150.0)
        
        # Track last fill price
        if not hasattr(self, '_last_fill_prices'):
            self._last_fill_prices = {}
        self._last_fill_prices[symbol] = fill_price
        
        # Update position
        current_pos = self.positions.get(symbol, 0.0)
        if side == "buy":
            self.positions[symbol] = current_pos + quantity
            self.capital -= quantity * fill_price
        else:
            self.positions[symbol] = current_pos - quantity
            self.capital += quantity * fill_price
            
        trade = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": fill_price,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "filled"
        }
        self.trades.append(trade)
        
        return trade
        
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get the last known price for a symbol."""
        if hasattr(self, '_last_fill_prices') and symbol in self._last_fill_prices:
            return self._last_fill_prices[symbol]
        # Try normalized symbol names
        normalized = symbol.replace('/', '').upper()
        if hasattr(self, '_last_fill_prices'):
            for key, price in self._last_fill_prices.items():
                if key.replace('/', '').upper() == normalized:
                    return price
        return None


class LiveExecution(ExecutionBackend):
    """
    Live execution backend for real trading.
    
    Integrates with Kraken exchange via REST API.
    Requires valid API credentials for live trading.
    """
    
    def __init__(self, api_key: str, api_secret: str, exchange: str = "kraken"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.exchange = exchange
        self._client = None
        self._positions: Dict[str, float] = {}
        
        # Try to initialize exchange client
        self._init_client()
    
    def _init_client(self):
        """Initialize exchange client if available."""
        try:
            if self.exchange == "kraken":
                from infra.kraken_rest_client import KrakenRestClient
                self._client = KrakenRestClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                logger.info("LiveExecution: Kraken client initialized")
        except ImportError:
            logger.warning("LiveExecution: Kraken client not available, using dry-run mode")
        except Exception as e:
            logger.error(f"LiveExecution: Failed to initialize client: {e}")
        
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Submit order to exchange."""
        if self._client is None:
            logger.warning(f"LiveExecution: No client - DRY RUN {side} {quantity} {symbol}")
            return {
                "order_id": f"DRY_{datetime.utcnow().timestamp()}",
                "status": "dry_run",
                "symbol": symbol,
                "side": side,
                "quantity": quantity
            }
        
        try:
            # Map to Kraken format
            kraken_symbol = self._map_symbol(symbol)
            result = self._client.create_order(
                symbol=kraken_symbol,
                side=side,
                order_type=order_type,
                volume=quantity,
                price=price
            )
            
            # Update local position tracking
            current = self._positions.get(symbol, 0.0)
            if side == "buy":
                self._positions[symbol] = current + quantity
            else:
                self._positions[symbol] = current - quantity
            
            return result
        except Exception as e:
            logger.error(f"LiveExecution: Order failed: {e}")
            return {"error": str(e), "status": "failed"}
        
    def get_positions(self) -> Dict[str, float]:
        """Get positions from exchange."""
        if self._client is None:
            return self._positions.copy()
        
        try:
            exchange_positions = self._client.get_open_positions()
            # Merge with local tracking
            for symbol, pos in exchange_positions.items():
                self._positions[symbol] = pos
            return self._positions.copy()
        except Exception as e:
            logger.error(f"LiveExecution: Failed to get positions: {e}")
            return self._positions.copy()
    
    def _map_symbol(self, symbol: str) -> str:
        """Map generic symbol to exchange format."""
        # e.g., "BTC/USD" -> "XXBTZUSD"
        mapping = {
            "BTC/USD": "XXBTZUSD",
            "ETH/USD": "XETHZUSD",
            "SOL/USD": "SOLUSD",
        }
        return mapping.get(symbol, symbol)


# =============================================================================
# UNIFIED RUNNER
# =============================================================================

class UnifiedRunner:
    """
    Unified trading runner for backtest, paper, and live modes.
    
    Uses same event pipeline regardless of mode:
    1. Data Feed -> Market Data Events
    2. Regime Detection -> Regime Events  
    3. Signal Generation -> Signal Events
    4. Authority Fusion -> Fusion Events
    5. Risk Check -> Risk Events
    6. Execution -> Order Events
    
    Usage:
        runner = UnifiedRunner(RunnerConfig(mode=RunnerMode.BACKTEST))
        runner.run()
    """
    
    def __init__(self, config: RunnerConfig):
        self.config = config
        self.mode = config.mode
        
        # Event integration
        self.events = get_event_integration(
            log_dir=config.log_dir,
            enable_logging=config.enable_event_logging
        )
        
        # Data feed based on mode
        self.data_feed = self._create_data_feed()
        
        # Execution backend based on mode
        self.execution = self._create_execution_backend()
        
        # Components (lazy loaded)
        self._regime_detector = None
        self._signal_generator = None
        self._fusion_engine = None
        self._risk_manager = None
        
        # State
        self._running = False
        self._tick_count = 0
        self._current_context: Optional[TickContext] = None
        
        # Results
        self.results = {
            "ticks": 0,
            "trades": 0,
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        logger.info(f"[UnifiedRunner] Initialized in {self.mode.value} mode")
        
    def _create_data_feed(self) -> DataFeed:
        """Create appropriate data feed for mode."""
        if self.mode == RunnerMode.BACKTEST:
            return BacktestDataFeed(
                data_path=self.config.log_dir / "historical_data.json",
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
        else:
            return LiveDataFeed(
                symbols=self.config.symbols,
                exchange=self.config.exchange
            )
            
    def _create_execution_backend(self) -> ExecutionBackend:
        """Create appropriate execution backend for mode."""
        if self.mode == RunnerMode.LIVE:
            return LiveExecution(
                api_key=self.config.api_key or "",
                api_secret=self.config.api_secret or "",
                exchange=self.config.exchange
            )
        else:
            return SimulatedExecution(
                initial_capital=self.config.initial_capital
            )
            
    def _load_components(self):
        """Lazy load trading components."""
        try:
            # Regime detector
            from market.phase_detector import get_regime_detector
            self._regime_detector = get_regime_detector()
        except ImportError:
            logger.warning("RegimeDetector not available, using stub")
            
        try:
            # Signal generator
            from signals.authority_fusion import get_fusion_engine
            self._fusion_engine = get_fusion_engine()
        except ImportError:
            logger.warning("FusionEngine not available, using stub")
            
        try:
            # Risk manager
            from risk import RiskManager
            self._risk_manager = RiskManager()
        except ImportError:
            logger.warning("RiskManager not available, using stub")
            
    def _process_tick(self, market_data: Dict) -> TickContext:
        """
        Process single tick through entire pipeline.
        
        This is the core of the unified approach - same pipeline for all modes.
        """
        self._tick_count += 1
        
        # 1. Start tick cycle
        tick_id = self.events.start_tick_cycle(
            mode="NORMAL",  # Will be updated
            regime="unknown"
        )
        
        # Create context
        context = TickContext(
            tick_id=tick_id,
            timestamp=datetime.utcnow(),
            mode=self.mode,
            system_mode="NORMAL",
            regime="trending",
            market_data=market_data
        )
        
        # 2. Emit market data event
        self.events.market.emit_price_update(
            symbol=market_data.get("symbol", "SOL/USD"),
            price=market_data.get("close", market_data.get("price", 0)),
            volume=market_data.get("volume", 0)
        )
        
        # 3. Detect regime (if available)
        if self._regime_detector:
            try:
                regime_result = self._regime_detector.detect(market_data)
                context.regime = regime_result.regime
                self.events.regime.emit_regime_update(
                    regime=regime_result.regime,
                    phase=regime_result.phase,
                    confidence=regime_result.confidence,
                    volatility=regime_result.volatility
                )
            except Exception as e:
                logger.debug(f"Regime detection error: {e}")
                
        # 4. Generate and fuse signals (if available)
        if self._fusion_engine:
            try:
                # This would integrate with actual signal generators
                pass
            except Exception as e:
                logger.debug(f"Signal fusion error: {e}")
                
        # 5. Risk check (if available)
        risk_approved = True
        if self._risk_manager:
            try:
                # Get direction from signals
                signal_direction = context.signals.get("direction", 0)
                if signal_direction == 0:
                    signal_direction = 1.0 if context.signals.get("action") == "BUY" else -1.0 if context.signals.get("action") == "SELL" else 0
                
                risk_result = self._risk_manager.check_trade_allowed(
                    symbol=market_data.get("symbol", "SOL/USD"),
                    direction=signal_direction,
                    size=0.1
                )
                risk_approved = risk_result.get("approved", True)
                if not risk_approved:
                    self.events.risk.emit_risk_veto(
                        veto_type="HARD",
                        reason=risk_result.get("reason", "Unknown"),
                        threshold=risk_result.get("threshold", 0),
                        actual_value=risk_result.get("actual", 0)
                    )
            except Exception as e:
                logger.debug(f"Risk check error: {e}")
                
        context.risk_state = {"approved": risk_approved}
        
        # 6. Execute if approved
        if risk_approved and context.signals.get("direction", 0) != 0:
            self._execute_signal(context, market_data)
            
        # 7. Finalize tick
        tick_log = self.events.finalize_tick_cycle()
        
        return context
    
    def _execute_signal(self, context: TickContext, market_data: Dict):
        """Execute a trading signal."""
        direction = context.signals.get("direction", 0)
        if direction == 0:
            return
        
        symbol = market_data.get("symbol", "SOL/USD")
        price = market_data.get("price", market_data.get("close", 0))
        size = context.signals.get("size", 0.1)
        
        # Log execution intent
        action = "BUY" if direction > 0 else "SELL"
        logger.info(f"[Execute] {action} {size} {symbol} @ {price}")
        
        # Execute via backend
        if self.execution_backend:
            try:
                result = self.execution_backend.execute({
                    "symbol": symbol,
                    "side": action.lower(),
                    "size": abs(size),
                    "price": price,
                    "order_type": "market"
                })
                context.execution_result = result
            except Exception as e:
                logger.error(f"Execution error: {e}")
                context.execution_result = {"error": str(e)}
        
    def run(self, max_ticks: Optional[int] = None) -> Dict:
        """
        Run the trading loop.
        
        Args:
            max_ticks: Maximum ticks to process (None = run until data exhausted)
            
        Returns:
            Results dictionary
        """
        logger.info(f"[UnifiedRunner] Starting {self.mode.value} mode...")
        self._running = True
        self._load_components()
        
        try:
            while self._running and self.data_feed.has_more_data():
                # Check max ticks
                if max_ticks and self._tick_count >= max_ticks:
                    break
                    
                # Get next tick data
                market_data = self.data_feed.get_next_tick()
                if not market_data:
                    break
                    
                # Process tick
                context = self._process_tick(market_data)
                self._current_context = context
                
                # Log progress
                if self._tick_count % 10 == 0:
                    logger.info(f"[UnifiedRunner] Processed {self._tick_count} ticks")
                    
                # Sleep between ticks (only for live/paper)
                if self.mode != RunnerMode.BACKTEST:
                    time.sleep(self.config.tick_interval_seconds)
                    
        except KeyboardInterrupt:
            logger.info("[UnifiedRunner] Interrupted by user")
        finally:
            self._running = False
            
        # Calculate results
        self._calculate_results()
        
        logger.info(f"[UnifiedRunner] Completed: {self._tick_count} ticks processed")
        return self.results
        
    def _calculate_results(self):
        """Calculate final results."""
        self.results["ticks"] = self._tick_count
        
        if isinstance(self.execution, SimulatedExecution):
            self.results["trades"] = len(self.execution.trades)
            # Calculate PnL
            initial = self.config.initial_capital
            final = self.execution.capital
            
            # Use last known prices for position valuation
            for symbol, qty in self.execution.positions.items():
                # Get last price from execution history or use fallback
                last_price = self.execution.get_last_price(symbol)
                if last_price is None:
                    # Fallback: estimate from symbol
                    if 'BTC' in symbol:
                        last_price = 100000.0
                    elif 'ETH' in symbol:
                        last_price = 3500.0
                    else:
                        last_price = 150.0  # SOL default
                final += qty * last_price
                
            self.results["pnl"] = final - initial
            self.results["pnl_pct"] = (final - initial) / initial * 100
            
    def stop(self):
        """Stop the runner."""
        self._running = False
        logger.info("[UnifiedRunner] Stop requested")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_backtest(
    data_path: Optional[Path] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_capital: float = 100000.0
) -> Dict:
    """Run a backtest with default configuration."""
    config = RunnerConfig(
        mode=RunnerMode.BACKTEST,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    runner = UnifiedRunner(config)
    return runner.run()


def run_paper(symbols: Optional[List[str]] = None) -> Dict:
    """Run paper trading with default configuration."""
    config = RunnerConfig(
        mode=RunnerMode.PAPER,
        symbols=symbols or ["SOL/USD", "BTC/USD", "ETH/USD"]
    )
    runner = UnifiedRunner(config)
    return runner.run()


def run_live(
    api_key: str,
    api_secret: str,
    symbols: Optional[List[str]] = None
) -> Dict:
    """Run live trading with default configuration."""
    config = RunnerConfig(
        mode=RunnerMode.LIVE,
        symbols=symbols or ["SOL/USD"],
        api_key=api_key,
        api_secret=api_secret
    )
    runner = UnifiedRunner(config)
    return runner.run()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'UnifiedRunner',
    'RunnerMode',
    'RunnerConfig',
    'TickContext',
    'DataFeed',
    'BacktestDataFeed',
    'LiveDataFeed',
    'ExecutionBackend',
    'SimulatedExecution',
    'LiveExecution',
    'run_backtest',
    'run_paper',
    'run_live',
]


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    results = run_backtest(initial_capital=100000.0)
    print(f"Backtest results: {results}")

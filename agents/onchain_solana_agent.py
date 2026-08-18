"""
================================================================================
HMATS v6.8.0 - On-Chain Solana Agent (ADVISE-only)
================================================================================
Purpose: Alpha signals from on-chain Solana behavior

Authority: ADVISE - modulates confidence, does NOT generate independent
           entry/exit decisions.  Fused via Authority Fusion weights.

Features:
1. DEX flow analysis (Birdeye API - volume + buy/sell ratio)
2. Whale activity tracking (Solscan API - activity intensity, NOT direction)
3. Exchange inflow/outflow (placeholder - marked mock)
4. Staking flow analysis (placeholder - marked mock)
5. Wallet activity metrics (placeholder - marked mock)

Production-readiness (v6.8):
- Every data source carries source_status {available|mock|error}
- Metrics include coverage map (real / inferred / mock / missing)
- Whale direction removed (was always 0); replaced with activity_score
- dex_volume_change fixed: now volume_mult vs rolling median (not price_change)
- generate_signal() uses all 4 weighted components incl. whale
- signal_type chosen by dominant contributor (not always DEX_FLOW_*)
- Fail-closed: mock/inferred inputs discount confidence

Author: HMATS SOTA Upgrade
Version: 6.8.0
================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


def _ensure_utc_timestamp(ts: datetime) -> datetime:
    """Coerce naive timestamps to UTC for legacy callers/tests."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OnChainSignalType(Enum):
    """Types of on-chain signals."""
    WHALE_BUY = "whale_buy"
    WHALE_SELL = "whale_sell"
    WHALE_ACTIVE = "whale_active"
    DEX_FLOW_BULLISH = "dex_flow_bullish"
    DEX_FLOW_BEARISH = "dex_flow_bearish"
    STAKING_INFLOW = "staking_inflow"
    STAKING_OUTFLOW = "staking_outflow"
    HIGH_ACTIVITY = "high_activity"
    LOW_ACTIVITY = "low_activity"
    TOKEN_LAUNCH = "token_launch"
    EXCHANGE_INFLOW = "exchange_inflow"    # Bearish
    EXCHANGE_OUTFLOW = "exchange_outflow"  # Bullish


# Explicit sentiment mapping for each signal type (+1 bullish, -1 bearish, 0 neutral)
_SIGNAL_SENTIMENT: Dict[OnChainSignalType, int] = {
    OnChainSignalType.WHALE_BUY: 1,
    OnChainSignalType.WHALE_SELL: -1,
    OnChainSignalType.WHALE_ACTIVE: 0,
    OnChainSignalType.DEX_FLOW_BULLISH: 1,
    OnChainSignalType.DEX_FLOW_BEARISH: -1,
    OnChainSignalType.STAKING_INFLOW: 1,
    OnChainSignalType.STAKING_OUTFLOW: -1,
    OnChainSignalType.HIGH_ACTIVITY: 0,
    OnChainSignalType.LOW_ACTIVITY: 0,
    OnChainSignalType.TOKEN_LAUNCH: 0,
    OnChainSignalType.EXCHANGE_INFLOW: -1,
    OnChainSignalType.EXCHANGE_OUTFLOW: 1,
}


@dataclass
class SourceStatus:
    """Status of a single data source."""
    name: str
    available: bool = False
    mock: bool = True
    error: str = ""
    last_ok_ts: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "available": self.available,
            "mock": self.mock,
            "error": self.error,
            "last_ok_ts": self.last_ok_ts.isoformat() if self.last_ok_ts else None,
        }


@dataclass
class OnChainMetrics:
    """Current on-chain metrics."""
    timestamp: datetime

    # DEX metrics
    dex_volume_24h: float = 0.0
    dex_volume_change: float = 0.0     # volume_mult vs median (1.0 = baseline)
    price_change_24h_pct: float = 0.0  # for debug only, NOT used as volume_spike
    buy_volume_pct: float = 0.5        # % of DEX volume that's buys

    # Wallet metrics
    active_wallets_24h: int = 0
    new_wallets_24h: int = 0
    whale_transactions: int = 0
    whale_activity_score: float = 0.0  # 0-1 intensity, NOT direction

    # Flow metrics
    exchange_net_flow: float = 0.0  # Positive = inflow (bearish)
    staking_net_flow: float = 0.0   # Positive = inflow (bullish)

    # Derived signals
    signals: List[OnChainSignalType] = field(default_factory=list)

    # Observability
    source_status: Dict[str, SourceStatus] = field(default_factory=dict)
    coverage: Dict[str, str] = field(default_factory=dict)  # field -> real|inferred|mock|missing

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "dex_volume_24h": self.dex_volume_24h,
            "dex_volume_change": self.dex_volume_change,
            "price_change_24h_pct": self.price_change_24h_pct,
            "buy_volume_pct": self.buy_volume_pct,
            "active_wallets_24h": self.active_wallets_24h,
            "new_wallets_24h": self.new_wallets_24h,
            "whale_transactions": self.whale_transactions,
            "whale_activity_score": self.whale_activity_score,
            "exchange_net_flow": self.exchange_net_flow,
            "staking_net_flow": self.staking_net_flow,
            "signals": [s.value for s in self.signals],
            "source_status": {k: v.to_dict() for k, v in self.source_status.items()},
            "coverage": self.coverage,
        }


@dataclass
class OnChainAlphaSignal:
    """Trading signal from on-chain analysis."""
    timestamp: datetime
    direction: float      # -1 to 1
    confidence: float     # 0 to 1
    signal_type: OnChainSignalType
    reasoning: str
    metrics: Dict = field(default_factory=dict)
    data_quality: float = 0.0     # 0-1 based on real vs mock coverage
    coverage: Dict = field(default_factory=dict)

    def to_agent_signal_dict(self) -> Dict:
        """Convert to flat dict for Authority Fusion / HMATS consumption.

        V6 contract: includes asof_timestamp, data_age_seconds, missing_inputs.
        """
        import time as _time
        data_age = self.metrics.get("data_age_seconds", 0)
        missing = [k for k, v in self.coverage.items() if v == "missing"]
        return {
            "onchain_direction": self.direction,
            "onchain_confidence": self.confidence,
            "onchain_signal_type": self.signal_type.value,
            "onchain_reasoning": self.reasoning,
            "onchain_data_quality": self.data_quality,
            "onchain_coverage": self.coverage,
            "missing_inputs": missing,
            "asof_timestamp": self.timestamp.timestamp() if hasattr(self.timestamp, 'timestamp') else 0.0,
            "data_age_seconds": data_age,
            "asset": "SOL",
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class OnChainConfig:
    """Configuration for on-chain agent."""
    # API Keys (set via environment or config)
    birdeye_api_key: str = ""  # Get from https://birdeye.so
    solscan_api_key: str = ""  # Get from https://solscan.io
    helius_api_key: str = ""   # Get from https://helius.dev

    # Data sources (public APIs)
    use_solscan: bool = True
    # [P307b] NO READER (with use_solscan below): declared, never
    # consulted. Neither provider is called anywhere in the tree, so
    # these describe an intended integration rather than a live switch.
    use_birdeye: bool = True
    use_flipside: bool = False

    # API endpoints
    birdeye_base_url: str = "https://public-api.birdeye.so"
    solscan_base_url: str = "https://api.solscan.io"

    # Thresholds
    whale_threshold_sol: float = 10000    # 10k SOL = whale
    volume_spike_threshold: float = 2.0    # 2x normal volume
    flow_significance_sol: float = 50000   # 50k SOL flow is significant

    # Update intervals
    update_interval_seconds: int = 300     # 5 minutes

    # Signal weights (must sum to ~1.0)
    dex_flow_weight: float = 0.3
    whale_weight: float = 0.25
    staking_weight: float = 0.2
    exchange_flow_weight: float = 0.25

    # Volume baseline: number of historical snapshots to compute median
    volume_baseline_window: int = 48  # ~4h at 5-min intervals


# =============================================================================
# ON-CHAIN DATA PROVIDER
# =============================================================================

class OnChainDataProvider:
    """
    Fetches on-chain data from various sources.

    Supported sources (free tiers):
    - Birdeye API (DEX data)
    - Solscan API (whale activity)
    - Exchange flows, staking, wallet: placeholders (marked mock)
    """

    def __init__(self, config: OnChainConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

        # Cache
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = create_session()
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()

    async def fetch_dex_metrics(self) -> Dict:
        """Fetch DEX volume and flow metrics from Birdeye API."""

        if not self.config.birdeye_api_key:
            logger.debug("Birdeye API key not configured, using mock data")
            return {
                "total_volume_24h": 0,
                "buy_volume": 0,
                "sell_volume": 0,
                "top_pairs": [],
                "price_change_24h_pct": 0,
                "_mock": True,
                "_source": "birdeye",
            }

        try:
            session = await self._get_session()

            sol_address = "So11111111111111111111111111111111111111112"
            headers = {
                "X-API-KEY": self.config.birdeye_api_key,
                "accept": "application/json",
            }
            url = f"{self.config.birdeye_base_url}/defi/token_overview?address={sol_address}"

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    token_data = data.get("data", {})

                    volume_24h = float(token_data.get("v24hUSD", 0))
                    price_change = float(token_data.get("v24hChangePercent", 0))

                    # Birdeye provides buy24h/sell24h on some endpoints;
                    # if absent, infer from price_change (clearly marked).
                    buy_24h = float(token_data.get("buy24h", 0))
                    sell_24h = float(token_data.get("sell24h", 0))

                    if buy_24h + sell_24h > 0:
                        buy_ratio = buy_24h / (buy_24h + sell_24h)
                        buy_source = "real"
                    elif volume_24h > 0:
                        # Fallback: infer from price_change (weak proxy)
                        buy_ratio = 0.5 + (max(-10, min(price_change, 10)) / 20)
                        buy_source = "inferred_from_price"
                    else:
                        buy_ratio = 0.5
                        buy_source = "default"

                    return {
                        "total_volume_24h": volume_24h,
                        "buy_volume": volume_24h * buy_ratio,
                        "sell_volume": volume_24h * (1 - buy_ratio),
                        "buy_source": buy_source,
                        "top_pairs": token_data.get("topPairs", [])[:5],
                        "price_change_24h_pct": price_change,
                        "_mock": False,
                        "_source": "birdeye",
                    }
                else:
                    logger.warning(f"Birdeye API returned {response.status}")

        except asyncio.TimeoutError:
            logger.warning("Birdeye API timeout")
        except Exception as e:
            logger.error(f"Birdeye API error: {e}")

        return {
            "total_volume_24h": 0,
            "buy_volume": 0,
            "sell_volume": 0,
            "buy_source": "error_fallback",
            "top_pairs": [],
            "price_change_24h_pct": 0,
            "_mock": True,
            "_source": "birdeye",
            "_error": "fetch_failed",
        }

    async def fetch_whale_activity(self) -> Dict:
        """Fetch recent whale transactions from Solscan API."""

        if not self.config.solscan_api_key:
            logger.debug("Solscan API key not configured, using mock data")
            return {
                "transactions": [],
                "total_volume": 0,
                "activity_score": 0.0,
                "_mock": True,
                "_source": "solscan",
            }

        try:
            session = await self._get_session()

            headers = {
                "token": self.config.solscan_api_key,
                "accept": "application/json",
            }
            url = (
                f"{self.config.solscan_base_url}/transfer/token"
                f"?token_address=So11111111111111111111111111111111111111112&limit=50"
            )

            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    transfers = data.get("data", [])

                    whale_txs = []
                    total_volume = 0.0

                    for tx in transfers:
                        amount = float(tx.get("amount", 0)) / 1e9  # lamports -> SOL
                        if amount >= self.config.whale_threshold_sol:
                            whale_txs.append({
                                "hash": tx.get("signature", ""),
                                "amount": amount,
                                "from": tx.get("src_address", ""),
                                "to": tx.get("dst_address", ""),
                                "time": tx.get("block_time", 0),
                            })
                            total_volume += amount

                    # Activity score: 0-1 based on how many whale txs in window.
                    # 10+ whale txs in a 5-min window => score 1.0
                    activity_score = min(1.0, len(whale_txs) / 10.0)

                    return {
                        "transactions": whale_txs[:10],
                        "total_volume": total_volume,
                        "activity_score": activity_score,
                        "_mock": False,
                        "_source": "solscan",
                    }
                else:
                    logger.warning(f"Solscan API returned {response.status}")

        except asyncio.TimeoutError:
            logger.warning("Solscan API timeout")
        except Exception as e:
            logger.error(f"Solscan API error: {e}")

        return {
            "transactions": [],
            "total_volume": 0,
            "activity_score": 0.0,
            "_mock": True,
            "_source": "solscan",
            "_error": "fetch_failed",
        }

    async def fetch_exchange_flows(self) -> Dict:
        """Fetch exchange inflow/outflow data. (Not implemented - mock)"""
        return {
            "inflow_24h": 0,
            "outflow_24h": 0,
            "net_flow": 0,
            "exchanges": {},
            "_mock": True,
            "_source": "exchange_flows",
        }

    async def fetch_staking_metrics(self) -> Dict:
        """Fetch staking/unstaking flows. (Not implemented - mock)"""
        return {
            "total_staked": 0,
            "staked_24h": 0,
            "unstaked_24h": 0,
            "net_flow": 0,
            "_mock": True,
            "_source": "staking",
        }

    async def fetch_wallet_activity(self) -> Dict:
        """Fetch wallet activity metrics. (Not implemented - mock)"""
        return {
            "active_wallets": 0,
            "new_wallets": 0,
            "transactions_24h": 0,
            "_mock": True,
            "_source": "wallet_activity",
        }


# =============================================================================
# SOLANA ON-CHAIN AGENT
# =============================================================================

class SolanaOnChainAgent:
    """
    ADVISE-only agent that analyzes on-chain Solana data.

    Authority: ADVISE - modulates confidence only; never generates
    independent entry/exit decisions.

    Signals generated:
    1. DEX flow imbalance (buy vs sell pressure)
    2. Whale activity intensity (NOT direction when unknown)
    3. Exchange flows (bullish when outflow) - currently mock
    4. Staking trends - currently mock
    5. Wallet activity trends - currently mock
    """

    def __init__(self, config: OnChainConfig = None):
        self.config = config or OnChainConfig()
        self.data_provider = OnChainDataProvider(self.config)

        # State
        self.current_metrics: Optional[OnChainMetrics] = None
        self.metrics_history: deque = deque(maxlen=288)  # 24h at 5min intervals

        # Rolling volume baseline for dex_volume_change
        self._volume_history: deque = deque(maxlen=self.config.volume_baseline_window)

        # Signal history
        self.signals: deque = deque(maxlen=100)

        # Background task
        self._update_task: Optional[asyncio.Task] = None
        self._running = False

        # V6: event emission callback (set by EventBus integration)
        self._event_callback = None

        logger.info("[SolanaOnChain] Agent initialized (ADVISE-only)")

    def set_event_callback(self, callback):
        """Set EventBus callback for ONCHAIN_TICK emission."""
        self._event_callback = callback

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self):
        """Start background data collection."""
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("[SolanaOnChain] Started background updates")

    async def stop(self):
        """Stop background data collection."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        await self.data_provider.close()
        logger.info("[SolanaOnChain] Stopped")

    async def _update_loop(self):
        """Background loop to update metrics."""
        while self._running:
            try:
                await self.update_metrics()
                await asyncio.sleep(self.config.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SolanaOnChain] Update error: {e}")
                await asyncio.sleep(60)

    # =========================================================================
    # METRICS COLLECTION
    # =========================================================================

    async def update_metrics(self):
        """Fetch and update all on-chain metrics."""
        results = await asyncio.gather(
            self.data_provider.fetch_dex_metrics(),
            self.data_provider.fetch_whale_activity(),
            self.data_provider.fetch_exchange_flows(),
            self.data_provider.fetch_staking_metrics(),
            self.data_provider.fetch_wallet_activity(),
            return_exceptions=True,
        )

        source_names = ["dex", "whale", "exchange_flows", "staking", "wallet_activity"]
        fetched: Dict[str, Dict] = {}
        source_statuses: Dict[str, SourceStatus] = {}

        for name, result in zip(source_names, results):
            if isinstance(result, BaseException):
                logger.warning(
                    f"[SolanaOnChain] Source '{name}' raised {type(result).__name__}: {result}"
                )
                source_statuses[name] = SourceStatus(
                    name=name, available=False, mock=True,
                    error=f"{type(result).__name__}: {result}",
                )
                fetched[name] = {}
            elif isinstance(result, dict):
                is_mock = result.get("_mock", True)
                has_error = bool(result.get("_error"))
                source_statuses[name] = SourceStatus(
                    name=name,
                    available=not is_mock,
                    mock=is_mock,
                    error=result.get("_error", ""),
                    last_ok_ts=datetime.now(timezone.utc) if (not is_mock and not has_error) else None,
                )
                fetched[name] = result
            else:
                logger.warning(f"[SolanaOnChain] Source '{name}' returned unexpected type: {type(result)}")
                source_statuses[name] = SourceStatus(
                    name=name, available=False, mock=True,
                    error=f"unexpected_type:{type(result).__name__}",
                )
                fetched[name] = {}

        # Build metrics
        metrics = OnChainMetrics(
            timestamp=datetime.now(timezone.utc),
            source_status=source_statuses,
        )
        coverage: Dict[str, str] = {}

        # --- DEX ---
        dex = fetched.get("dex", {})
        if dex:
            metrics.dex_volume_24h = float(dex.get("total_volume_24h", 0))
            metrics.price_change_24h_pct = float(dex.get("price_change_24h_pct", 0))

            # Volume mult vs rolling median
            if metrics.dex_volume_24h > 0:
                self._volume_history.append(metrics.dex_volume_24h)
            if len(self._volume_history) >= 2:
                sorted_vols = sorted(self._volume_history)
                median_vol = sorted_vols[len(sorted_vols) // 2]
                if median_vol > 0:
                    metrics.dex_volume_change = metrics.dex_volume_24h / median_vol
                    coverage["dex_volume_change"] = "real"
                else:
                    metrics.dex_volume_change = 1.0
                    coverage["dex_volume_change"] = "missing"
            else:
                metrics.dex_volume_change = 1.0  # not enough baseline
                coverage["dex_volume_change"] = "missing"

            total_vol = float(dex.get("buy_volume", 0)) + float(dex.get("sell_volume", 0))
            if total_vol > 0:
                metrics.buy_volume_pct = float(dex.get("buy_volume", 0)) / total_vol
            buy_source = dex.get("buy_source", "default")
            coverage["buy_volume_pct"] = buy_source if buy_source in ("real",) else "inferred"
            coverage["dex_volume_24h"] = "mock" if dex.get("_mock") else "real"
        else:
            coverage["dex_volume_24h"] = "missing"
            coverage["buy_volume_pct"] = "missing"
            coverage["dex_volume_change"] = "missing"

        # --- Whale ---
        whale = fetched.get("whale", {})
        if whale:
            metrics.whale_transactions = len(whale.get("transactions", []))
            metrics.whale_activity_score = float(whale.get("activity_score", 0.0))
            coverage["whale_activity_score"] = "mock" if whale.get("_mock") else "real"
        else:
            coverage["whale_activity_score"] = "missing"

        # --- Exchange flows ---
        exch = fetched.get("exchange_flows", {})
        if exch:
            metrics.exchange_net_flow = float(exch.get("net_flow", 0))
            coverage["exchange_net_flow"] = "mock" if exch.get("_mock") else "real"
        else:
            coverage["exchange_net_flow"] = "missing"

        # --- Staking ---
        stak = fetched.get("staking", {})
        if stak:
            metrics.staking_net_flow = float(stak.get("net_flow", 0))
            coverage["staking_net_flow"] = "mock" if stak.get("_mock") else "real"
        else:
            coverage["staking_net_flow"] = "missing"

        # --- Wallet ---
        wall = fetched.get("wallet_activity", {})
        if wall:
            metrics.active_wallets_24h = int(wall.get("active_wallets", 0))
            metrics.new_wallets_24h = int(wall.get("new_wallets", 0))
            coverage["active_wallets_24h"] = "mock" if wall.get("_mock") else "real"
        else:
            coverage["active_wallets_24h"] = "missing"

        metrics.coverage = coverage

        # Derive signals
        metrics.signals = self._derive_signals(metrics)

        # Store
        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        logger.debug(
            f"[SolanaOnChain] Updated metrics: {len(metrics.signals)} signals, "
            f"sources={{{', '.join(f'{k}:{v.available}' for k, v in source_statuses.items())}}}"
        )

        # V6: emit ONCHAIN_TICK event
        if self._event_callback:
            try:
                self._event_callback("ONCHAIN_TICK", {
                    "asset": "SOL",
                    **metrics.to_dict(),
                })
            except Exception as e:
                logger.debug(f"[SolanaOnChain] Event emission failed: {e}")

    def _derive_signals(self, metrics: OnChainMetrics) -> List[OnChainSignalType]:
        """Derive trading signals from metrics."""
        signals = []

        # DEX flow signal (only if coverage is real or inferred, not missing)
        cov_buy = metrics.coverage.get("buy_volume_pct", "missing")
        if cov_buy != "missing":
            if metrics.buy_volume_pct > 0.6:
                signals.append(OnChainSignalType.DEX_FLOW_BULLISH)
            elif metrics.buy_volume_pct < 0.4:
                signals.append(OnChainSignalType.DEX_FLOW_BEARISH)

        # Volume spike (only with enough baseline)
        cov_vol = metrics.coverage.get("dex_volume_change", "missing")
        if cov_vol == "real":
            if metrics.dex_volume_change > self.config.volume_spike_threshold:
                signals.append(OnChainSignalType.HIGH_ACTIVITY)
            elif metrics.dex_volume_change < 0.5:
                signals.append(OnChainSignalType.LOW_ACTIVITY)

        # Whale activity (intensity, not direction)
        if metrics.whale_activity_score > 0.5:
            signals.append(OnChainSignalType.WHALE_ACTIVE)

        # Exchange flows (only if not mock)
        cov_exch = metrics.coverage.get("exchange_net_flow", "missing")
        if cov_exch == "real":
            if metrics.exchange_net_flow > self.config.flow_significance_sol:
                signals.append(OnChainSignalType.EXCHANGE_INFLOW)
            elif metrics.exchange_net_flow < -self.config.flow_significance_sol:
                signals.append(OnChainSignalType.EXCHANGE_OUTFLOW)

        # Staking flows (only if not mock)
        cov_stak = metrics.coverage.get("staking_net_flow", "missing")
        if cov_stak == "real":
            if metrics.staking_net_flow > self.config.flow_significance_sol:
                signals.append(OnChainSignalType.STAKING_INFLOW)
            elif metrics.staking_net_flow < -self.config.flow_significance_sol:
                signals.append(OnChainSignalType.STAKING_OUTFLOW)

        return signals

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def generate_signal(self) -> Optional[OnChainAlphaSignal]:
        """
        Generate ADVISE trading signal from current on-chain state.

        Returns:
            OnChainAlphaSignal or None if no clear signal
        """
        if self.current_metrics is None:
            return None

        metrics = self.current_metrics

        # Component scores: (score, weight, name, coverage_quality)
        components: List[tuple] = []

        # 1) DEX flow score
        # [FIX-AG6] Only use real buy/sell data. Inferred-from-price is lagging (buys on rallies, sells on dumps).
        cov_buy = metrics.coverage.get("buy_volume_pct", "missing")
        if cov_buy == "real":
            dex_raw = (metrics.buy_volume_pct - 0.5) * 2  # -1 to 1
            components.append((dex_raw, self.config.dex_flow_weight, "dex", cov_buy))
        elif cov_buy == "inferred":
            # Heavily discount inferred data (price-as-proxy is lagging indicator)
            dex_raw = (metrics.buy_volume_pct - 0.5) * 2
            components.append((dex_raw, self.config.dex_flow_weight * 0.2, "dex_inferred", cov_buy))

        # 2) Exchange flow score (outflow = bullish)
        cov_exch = metrics.coverage.get("exchange_net_flow", "missing")
        if cov_exch == "real" and metrics.exchange_net_flow != 0:
            flow_mag = min(abs(metrics.exchange_net_flow) / self.config.flow_significance_sol, 1.0)
            exchange_raw = -flow_mag if metrics.exchange_net_flow > 0 else flow_mag
            components.append((exchange_raw, self.config.exchange_flow_weight, "exchange", cov_exch))

        # 3) Staking flow score (inflow = bullish)
        cov_stak = metrics.coverage.get("staking_net_flow", "missing")
        if cov_stak == "real" and metrics.staking_net_flow != 0:
            flow_mag = min(abs(metrics.staking_net_flow) / self.config.flow_significance_sol, 1.0)
            staking_raw = flow_mag if metrics.staking_net_flow > 0 else -flow_mag
            components.append((staking_raw, self.config.staking_weight, "staking", cov_stak))

        # 4) Whale activity score (intensity amplifier, not directional)
        cov_whale = metrics.coverage.get("whale_activity_score", "missing")
        if cov_whale != "missing" and metrics.whale_activity_score > 0.1:
            # Whale activity amplifies existing signal direction but doesn't
            # create direction on its own. Use sign of current sum as base.
            components.append((0.0, self.config.whale_weight, "whale", cov_whale))

        if not components:
            return None

        # Weighted sum
        direction = 0.0
        reasons = []
        dominant_component = ("", 0.0)

        for raw_score, weight, name, cov in components:
            weighted = raw_score * weight
            direction += weighted

            if name == "whale":
                # Skip reasoning for whale at zero, handled below
                continue
            if abs(weighted) > 0.02:
                label = {
                    "dex": f"DEX {'buy' if raw_score > 0 else 'sell'} pressure ({metrics.buy_volume_pct:.0%})",
                    "exchange": f"Exchange {'inflow (bearish)' if raw_score < 0 else 'outflow (bullish)'}",
                    "staking": f"Staking {'inflow (bullish)' if raw_score > 0 else 'outflow (bearish)'}",
                }.get(name, name)
                if cov != "real":
                    label += f" [[WARN] {cov}]"
                reasons.append(label)
                if abs(weighted) > abs(dominant_component[1]):
                    dominant_component = (name, weighted)

        # Whale amplification: scale signal magnitude by whale_activity_score
        if metrics.whale_activity_score > 0.1 and cov_whale != "missing":
            amplifier = 1.0 + (metrics.whale_activity_score * self.config.whale_weight)
            direction *= amplifier
            if metrics.whale_activity_score > 0.3:
                reasons.append(f"Whale activity={metrics.whale_activity_score:.2f} (amplifier)")

        direction = max(-1.0, min(1.0, direction))

        # Confidence: base from signal strength + penalties
        confidence = min(0.8, abs(direction) + 0.2)

        # Penalty for inferred/mock inputs
        inferred_count = sum(1 for v in metrics.coverage.values() if v in ("inferred", "mock"))
        total_count = max(len(metrics.coverage), 1)
        mock_ratio = inferred_count / total_count
        confidence *= (1.0 - 0.4 * mock_ratio)  # Up to 40% discount

        # Age penalty
        metrics_ts = _ensure_utc_timestamp(metrics.timestamp)
        data_age_seconds = (datetime.now(timezone.utc) - metrics_ts).total_seconds()
        if data_age_seconds > 600:
            confidence *= 0.8
        if data_age_seconds > 1800:
            confidence *= 0.5

        confidence = max(0.0, min(1.0, confidence))

        # Dead zone
        if abs(direction) < 0.05:
            return None

        # Determine signal_type from dominant contributor
        dom_name = dominant_component[0]
        if direction > 0:
            signal_type = {
                "dex": OnChainSignalType.DEX_FLOW_BULLISH,
                "exchange": OnChainSignalType.EXCHANGE_OUTFLOW,
                "staking": OnChainSignalType.STAKING_INFLOW,
            }.get(dom_name, OnChainSignalType.DEX_FLOW_BULLISH)
        else:
            signal_type = {
                "dex": OnChainSignalType.DEX_FLOW_BEARISH,
                "exchange": OnChainSignalType.EXCHANGE_INFLOW,
                "staking": OnChainSignalType.STAKING_OUTFLOW,
            }.get(dom_name, OnChainSignalType.DEX_FLOW_BEARISH)

        # Compute data_quality: ratio of real sources to total
        real_count = sum(1 for v in metrics.coverage.values() if v == "real")
        total_count = max(len(metrics.coverage), 1)
        signal_data_quality = real_count / total_count

        signal = OnChainAlphaSignal(
            timestamp=datetime.now(timezone.utc),
            direction=direction,
            confidence=confidence,
            signal_type=signal_type,
            reasoning="; ".join(reasons) if reasons else "Mixed signals",
            metrics={
                **metrics.to_dict(),
                "data_age_seconds": data_age_seconds,
            },
            data_quality=round(signal_data_quality, 3),
            coverage=dict(metrics.coverage),
        )

        self.signals.append(signal)
        return signal

    def generate_signal_safe(self) -> OnChainAlphaSignal:
        """
        HMATS-compatible wrapper: always returns signal (never None, never raises).

        When no signal is available, returns neutral with is_valid-like zero confidence.
        """
        try:
            signal = self.generate_signal()
            if signal is not None:
                return signal
        except Exception as e:
            logger.error(f"[SolanaOnChain] generate_signal_safe failed: {e}")

        return OnChainAlphaSignal(
            timestamp=datetime.now(timezone.utc),
            direction=0.0,
            confidence=0.0,
            signal_type=OnChainSignalType.LOW_ACTIVITY,
            reasoning="no_data_or_no_signal",
            data_quality=0.0,
        )

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_current_metrics(self) -> Optional[Dict]:
        """Get current on-chain metrics."""
        if self.current_metrics is None:
            return None
        d = self.current_metrics.to_dict()
        current_ts = _ensure_utc_timestamp(self.current_metrics.timestamp)
        d["data_age_seconds"] = (datetime.now(timezone.utc) - current_ts).total_seconds()
        return d

    def get_recent_signals(self, count: int = 10) -> List[Dict]:
        """Get recent signals."""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "direction": s.direction,
                "confidence": s.confidence,
                "type": s.signal_type.value,
                "reasoning": s.reasoning,
            }
            for s in list(self.signals)[-count:]
        ]

    def get_metrics_summary(self) -> Dict:
        """Get summary of on-chain state."""
        if not self.current_metrics:
            return {"status": "no_data"}

        m = self.current_metrics

        # Use explicit mapping, not string matching
        bullish_signals = sum(1 for s in m.signals if _SIGNAL_SENTIMENT.get(s, 0) > 0)
        bearish_signals = sum(1 for s in m.signals if _SIGNAL_SENTIMENT.get(s, 0) < 0)

        if bullish_signals > bearish_signals:
            sentiment = "bullish"
        elif bearish_signals > bullish_signals:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        mock_sources = [k for k, v in m.source_status.items() if v.mock]

        return {
            "status": "ok",
            "sentiment": sentiment,
            "dex_volume_24h": m.dex_volume_24h,
            "buy_pressure": m.buy_volume_pct,
            "whale_activity_score": m.whale_activity_score,
            "exchange_net_flow": m.exchange_net_flow,
            "staking_net_flow": m.staking_net_flow,
            "active_signals": [s.value for s in m.signals],
            "mock_sources": mock_sources,
            "data_age_seconds": (datetime.now(timezone.utc) - _ensure_utc_timestamp(m.timestamp)).total_seconds(),
        }


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

# agents/__init__.py and orchestration/sota_integration.py import
# "OnChainSolanaAgent" but the canonical class is SolanaOnChainAgent.
OnChainSolanaAgent = SolanaOnChainAgent


# =============================================================================
# SINGLETON
# =============================================================================

_onchain_agent: Optional[SolanaOnChainAgent] = None


def get_onchain_agent(config: OnChainConfig = None) -> SolanaOnChainAgent:
    """Get or create the global on-chain agent."""
    global _onchain_agent
    if _onchain_agent is None:
        _onchain_agent = SolanaOnChainAgent(config)
    elif config is not None and _onchain_agent.config != config:
        logger.info("[ONCHAIN_SOLANA] Config changed; refreshing singleton instance")
        _onchain_agent = SolanaOnChainAgent(config)
    return _onchain_agent


def reset_onchain_agent():
    """Reset the global singleton (for testing)."""
    global _onchain_agent
    _onchain_agent = None


async def shutdown_onchain_agent():
    """Shutdown the global on-chain agent."""
    global _onchain_agent
    if _onchain_agent:
        await _onchain_agent.stop()
    _onchain_agent = None

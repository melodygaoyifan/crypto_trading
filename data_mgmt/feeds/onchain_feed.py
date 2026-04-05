"""
================================================================================
HMATS v5.2.1 P0 - 链上数据采集器
On-Chain Data Feed
================================================================================

负责从链上数据源采集数据，输出标准化 OnChainTick。

功能:
1. 鲸鱼地址活动监控
2. 交易所流入流出
3. DEX 交易量和 TVL
4. 智能合约交互

生产替换说明:
- 当前使用 mock/placeholder API
- 生产环境替换 _fetch_from_source() 中的实现
- 支持 Solana RPC / The Graph / Dune Analytics 等

================================================================================
"""

import logging
import asyncio
import aiohttp
from data_mgmt.feeds._http import create_session
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import random

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class OnChainDataSource(Enum):
    """链上数据源"""
    SOLANA_RPC = "solana_rpc"
    THE_GRAPH = "the_graph"
    DUNE = "dune"
    HELIUS = "helius"
    COMPOSITE = "composite"  # CryptoCompare (BTC/ETH) + Solana RPC + Jito (SOL)
    MOCK = "mock"


@dataclass
class WhaleActivity:
    """鲸鱼活动"""
    address: str
    activity_type: str  # buy, sell, transfer
    token: str
    amount: float
    value_usd: float
    timestamp: datetime
    tx_hash: str = ""


@dataclass
class ExchangeFlow:
    """交易所流入流出"""
    exchange: str
    token: str
    direction: str  # inflow, outflow
    amount: float
    value_usd: float
    timestamp: datetime


@dataclass
class DEXMetrics:
    """DEX 指标"""
    protocol: str
    token: str
    volume_24h: float
    tvl: float
    trades_24h: int
    unique_traders: int
    timestamp: datetime


@dataclass
class OnChainTick:
    """
    标准化链上数据输出
    
    所有链上数据必须通过此格式输出
    """
    timestamp: datetime
    staleness_sec: float
    confidence: float  # 0-1, 数据可信度
    
    # 数据内容
    whale_activities: List[WhaleActivity] = field(default_factory=list)
    exchange_flows: List[ExchangeFlow] = field(default_factory=list)
    dex_metrics: List[DEXMetrics] = field(default_factory=list)
    
    # 聚合指标
    net_whale_flow_usd: float = 0.0  # 正=买入，负=卖出
    net_exchange_flow_usd: float = 0.0  # 正=流入，负=流出
    total_dex_volume_24h: float = 0.0
    
    # 元数据
    source: str = "unknown"
    tokens_covered: List[str] = field(default_factory=list)
    
    def is_stale(self, max_staleness_sec: float = 300) -> bool:
        """检查数据是否过期"""
        return self.staleness_sec > max_staleness_sec
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "staleness_sec": self.staleness_sec,
            "confidence": self.confidence,
            "net_whale_flow_usd": self.net_whale_flow_usd,
            "net_exchange_flow_usd": self.net_exchange_flow_usd,
            "total_dex_volume_24h": self.total_dex_volume_24h,
            "source": self.source,
            "tokens_covered": self.tokens_covered,
            "whale_count": len(self.whale_activities),
            "exchange_flow_count": len(self.exchange_flows),
            "dex_metric_count": len(self.dex_metrics),
        }


# =============================================================================
# ONCHAIN FEED
# =============================================================================

class OnChainFeed:
    """
    链上数据采集器
    
    生产级数据管道，所有链上数据采集必须通过此类。
    禁止在 Agent 内直接调用链上 API。
    """
    
    def __init__(
        self,
        source: OnChainDataSource = OnChainDataSource.MOCK,
        tokens: List[str] = None,
        poll_interval_sec: float = 60,
        event_bus_callback: Optional[Callable] = None,
    ):
        self.source = source
        self.tokens = tokens or ["SOL", "BTC", "ETH"]
        self.poll_interval_sec = poll_interval_sec
        self._event_bus_callback = event_bus_callback
        
        # 内部状态
        self._last_tick: Optional[OnChainTick] = None
        self._last_fetch_time: Optional[datetime] = None
        self._running = False
        self._fetch_errors = 0
        self._total_fetches = 0
        
        # 配置
        self._whale_threshold_usd = 100_000
        self._staleness_threshold_sec = 300
        
        logger.info(f"OnChainFeed initialized: source={source.value}, tokens={self.tokens}")
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    async def start(self):
        """启动数据采集"""
        if self._running:
            return
        
        self._running = True
        logger.info("OnChainFeed started")
        
        while self._running:
            try:
                tick = await self.fetch()
                if tick and self._event_bus_callback:
                    self._event_bus_callback("ONCHAIN_TICK", tick.to_dict())
            except Exception as e:
                logger.error(f"OnChainFeed fetch error: {e}")
                self._fetch_errors += 1
            
            await asyncio.sleep(self.poll_interval_sec)
    
    def stop(self):
        """停止数据采集"""
        self._running = False
        logger.info("OnChainFeed stopped")
    
    async def fetch(self) -> Optional[OnChainTick]:
        """
        获取最新链上数据
        
        Returns:
            OnChainTick: 标准化链上数据
        """
        self._total_fetches += 1
        fetch_start = datetime.now()
        
        try:
            # 从数据源获取原始数据
            raw_data = await self._fetch_from_source()
            
            # 转换为标准格式
            tick = self._parse_raw_data(raw_data)
            
            # 计算 staleness
            tick.staleness_sec = (datetime.now() - tick.timestamp).total_seconds()
            
            # 缓存
            self._last_tick = tick
            self._last_fetch_time = datetime.now()
            
            return tick
            
        except Exception as e:
            logger.error(f"OnChainFeed fetch failed: {e}")
            self._fetch_errors += 1
            return None
    
    def get_latest(self) -> Optional[OnChainTick]:
        """
        获取最新缓存的数据（非阻塞）
        
        Returns:
            OnChainTick: 最新缓存数据，如果过期则更新 staleness
        """
        if self._last_tick:
            # 更新 staleness
            self._last_tick.staleness_sec = (
                datetime.now() - self._last_tick.timestamp
            ).total_seconds()
        return self._last_tick
    
    def get_health(self) -> Dict[str, Any]:
        """获取数据健康状态"""
        tick = self.get_latest()
        
        return {
            "source": self.source.value,
            "last_fetch_time": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
            "staleness_sec": tick.staleness_sec if tick else float('inf'),
            "is_stale": tick.is_stale(self._staleness_threshold_sec) if tick else True,
            "confidence": tick.confidence if tick else 0.0,
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "error_rate": self._fetch_errors / max(1, self._total_fetches),
            "tokens_covered": self.tokens,
        }
    
    # =========================================================================
    # DATA SOURCE (生产环境需替换)
    # =========================================================================
    
    async def _fetch_from_source(self) -> Dict[str, Any]:
        """
        从数据源获取原始数据
        
        生产替换说明:
        - SOLANA_RPC: 替换为 solana-py RPC 调用
        - THE_GRAPH: 替换为 GraphQL 查询
        - DUNE: 替换为 Dune API 调用
        - HELIUS: 替换为 Helius API 调用
        """
        if self.source == OnChainDataSource.COMPOSITE:
            return await self._fetch_composite()

        if self.source == OnChainDataSource.MOCK:
            return await self._fetch_mock()

        if self.source == OnChainDataSource.HELIUS:
            return await self._fetch_helius()
        
        if self.source == OnChainDataSource.SOLANA_RPC:
            return await self._fetch_solana_rpc()
        
        logger.warning(f"Unknown source {self.source}, using mock")
        return await self._fetch_mock()
    
    async def _fetch_composite(self) -> Dict[str, Any]:
        """
        Fetch real on-chain data from CryptoCompare (BTC/ETH) + Solana RPC (SOL).
        Maps data into the dict format expected by _parse_raw_data().
        Falls back to mock if all real sources fail.
        """
        now = datetime.now()
        whale_activities = []
        exchange_flows = []
        dex_metrics = []
        any_real = False

        # 1. CryptoCompare: BTC + ETH on-chain metrics
        try:
            from data_mgmt.feeds.cryptocompare_onchain import get_cryptocompare_onchain_feed
            cc_feed = get_cryptocompare_onchain_feed()
            cc_data = await cc_feed.fetch()

            for symbol in ("BTC", "ETH"):
                cc = cc_data.get(symbol)
                if cc and not cc.is_mock and cc.large_transaction_count > 0:
                    any_real = True
                    # Large transactions as whale proxy
                    # Estimate volume: large_tx_count * avg_tx_value * rough_price
                    # avg_tx_value is in native units; we don't have price here,
                    # so use count as the signal (downstream normalizes)
                    whale_activities.append({
                        "address": f"cc_{symbol}_aggregate",
                        "activity_type": "buy",  # net direction unknown, neutral
                        "token": symbol,
                        "amount": float(cc.large_transaction_count),
                        "value_usd": float(cc.large_transaction_count) * cc.average_transaction_value,
                        "timestamp": cc.timestamp.isoformat(),
                        "tx_hash": "",
                    })

        except Exception as e:
            logger.warning(f"[ONCHAIN_COMPOSITE] CryptoCompare failed: {e}")

        # 2. Solana RPC: SOL network health + Jito MEV
        try:
            from data_mgmt.feeds.solana_onchain import get_solana_onchain_feed
            sol_feed = get_solana_onchain_feed()
            sol_data = await sol_feed.fetch()

            if not sol_data.is_mock:
                any_real = True
                # Map Jito pressure as a DEX-like metric for SOL
                if sol_data.jito_tip_50th_sol > 0:
                    dex_metrics.append({
                        "protocol": "jito",
                        "token": "SOL",
                        "volume_24h": 0.0,  # not available from tip endpoint
                        "tvl": 0.0,
                        "trades_24h": 0,
                        "unique_traders": 0,
                        "timestamp": sol_data.timestamp.isoformat(),
                        # Extra fields for downstream (ignored by _parse_raw_data
                        # but available via raw dict)
                    })

        except Exception as e:
            logger.warning(f"[ONCHAIN_COMPOSITE] Solana RPC failed: {e}")

        if not any_real:
            logger.debug("[ONCHAIN_COMPOSITE] No real data, falling back to mock")
            return await self._fetch_mock()

        return {
            "timestamp": now.isoformat(),
            "whale_activities": whale_activities,
            "exchange_flows": exchange_flows,
            "dex_metrics": dex_metrics,
            "confidence": 0.6,  # Real data but incomplete (no exchange flows)
            "_source": "composite",
            "_mock": False,
        }

    async def _fetch_helius(self) -> Dict[str, Any]:
        """Fetch from Helius API (requires API key)."""
        api_key = getattr(self.config, 'helius_api_key', '')
        if not api_key:
            logger.debug("Helius API key not configured, using mock")
            return await self._fetch_mock()
        
        try:
            async with create_session() as session:
                # SOL token mint address
                sol_mint = "So11111111111111111111111111111111111111112"
                
                # Fetch token holder stats (whales)
                url = f"https://api.helius.xyz/v0/token-metadata?api-key={api_key}"
                payload = {"mintAccounts": [sol_mint]}
                
                whale_activities = []
                exchange_flows = []
                dex_volumes = []
                
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Process token metadata
                        token_data = data[0] if data else {}
                        
                        # For whale activity, we'd need to query recent large transfers
                        # This requires the Helius Enhanced Transactions API
                        tx_url = f"https://api.helius.xyz/v0/addresses/{sol_mint}/transactions?api-key={api_key}&limit=10"
                        async with session.get(tx_url, timeout=10) as tx_response:
                            if tx_response.status == 200:
                                tx_data = await tx_response.json()
                                for tx in tx_data[:5]:
                                    # Parse large transfers as whale activity
                                    if tx.get('nativeTransfers'):
                                        for transfer in tx['nativeTransfers']:
                                            amount = transfer.get('amount', 0) / 1e9  # lamports to SOL
                                            if amount > 10000:  # Whale threshold
                                                whale_activities.append({
                                                    "address": transfer.get('fromUserAccount', '')[:16],
                                                    "activity_type": "transfer",
                                                    "token": "SOL",
                                                    "amount": amount,
                                                    "value_usd": amount * 150,  # Rough estimate
                                                    "timestamp": datetime.now().isoformat(),
                                                    "tx_hash": tx.get('signature', '')
                                                })
                        
                        return {
                            "whale_activities": whale_activities,
                            "exchange_flows": exchange_flows,
                            "dex_volumes": dex_volumes,
                            "smart_contract_activities": [],
                            "_source": "helius",
                            "_mock": False
                        }
                        
        except asyncio.TimeoutError:
            logger.warning("Helius API timeout")
        except Exception as e:
            logger.error(f"Helius API error: {e}")
        
        return await self._fetch_mock()
    
    async def _fetch_solana_rpc(self) -> Dict[str, Any]:
        """Fetch from Solana RPC (public endpoint or custom)."""
        rpc_url = getattr(self.config, 'solana_rpc_url', 'https://api.mainnet-beta.solana.com')
        
        try:
            async with create_session() as session:
                # Get recent block for activity metrics
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getRecentBlockhash"
                }
                
                async with session.post(rpc_url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Basic on-chain health indicator
                        return {
                            "whale_activities": [],
                            "exchange_flows": [],
                            "dex_volumes": [],
                            "smart_contract_activities": [],
                            "network_health": {
                                "blockhash": data.get('result', {}).get('value', {}).get('blockhash', ''),
                                "slot": data.get('result', {}).get('context', {}).get('slot', 0)
                            },
                            "_source": "solana_rpc",
                            "_mock": False
                        }
                        
        except Exception as e:
            logger.error(f"Solana RPC error: {e}")
        
        return await self._fetch_mock()
    
    async def _fetch_mock(self) -> Dict[str, Any]:
        """
        Neutral-safe fallback when on-chain feeds are unavailable.

        [PATCH-7c] Replaced random data with empty/neutral constants.
        Root cause: random mock data injected fake whale activity, exchange flows,
        and DEX metrics into agent signals on feed failure. Downstream agents
        (whale detector, on-chain alpha) would act on noise.
        """
        await asyncio.sleep(0.1)

        now = datetime.now()

        # No mock whale activity — empty list means "no data"
        whale_activities = []

        # No mock exchange flows — empty list means "no data"
        exchange_flows = []

        # No mock DEX metrics — empty list means "no data"
        dex_metrics = []

        return {
            "timestamp": now.isoformat(),
            "whale_activities": whale_activities,
            "exchange_flows": exchange_flows,
            "dex_metrics": dex_metrics,
            "confidence": 0.0,  # Zero confidence: downstream should ignore mock data
        }
    
    def _parse_raw_data(self, raw: Dict[str, Any]) -> OnChainTick:
        """解析原始数据为标准格式"""
        timestamp = datetime.fromisoformat(raw.get("timestamp", datetime.now().isoformat()))
        
        # 解析鲸鱼活动
        whale_activities = []
        for w in raw.get("whale_activities", []):
            whale_activities.append(WhaleActivity(
                address=w["address"],
                activity_type=w["activity_type"],
                token=w["token"],
                amount=w["amount"],
                value_usd=w["value_usd"],
                timestamp=datetime.fromisoformat(w["timestamp"]),
                tx_hash=w.get("tx_hash", ""),
            ))
        
        # 解析交易所流
        exchange_flows = []
        for f in raw.get("exchange_flows", []):
            exchange_flows.append(ExchangeFlow(
                exchange=f["exchange"],
                token=f["token"],
                direction=f["direction"],
                amount=f["amount"],
                value_usd=f["value_usd"],
                timestamp=datetime.fromisoformat(f["timestamp"]),
            ))
        
        # 解析 DEX 指标
        dex_metrics = []
        for d in raw.get("dex_metrics", []):
            dex_metrics.append(DEXMetrics(
                protocol=d["protocol"],
                token=d["token"],
                volume_24h=d["volume_24h"],
                tvl=d["tvl"],
                trades_24h=d["trades_24h"],
                unique_traders=d["unique_traders"],
                timestamp=datetime.fromisoformat(d["timestamp"]),
            ))
        
        # 计算聚合指标
        net_whale_flow = sum(
            w.value_usd if w.activity_type == "buy" else -w.value_usd
            for w in whale_activities
        )
        
        net_exchange_flow = sum(
            f.value_usd if f.direction == "inflow" else -f.value_usd
            for f in exchange_flows
        )
        
        total_dex_volume = sum(d.volume_24h for d in dex_metrics)
        
        return OnChainTick(
            timestamp=timestamp,
            staleness_sec=0.0,
            confidence=raw.get("confidence", 0.8),
            whale_activities=whale_activities,
            exchange_flows=exchange_flows,
            dex_metrics=dex_metrics,
            net_whale_flow_usd=net_whale_flow,
            net_exchange_flow_usd=net_exchange_flow,
            total_dex_volume_24h=total_dex_volume,
            source=self.source.value,
            tokens_covered=self.tokens,
        )


# =============================================================================
# SINGLETON
# =============================================================================

_onchain_feed_instance: Optional[OnChainFeed] = None


def get_onchain_feed(
    source: OnChainDataSource = OnChainDataSource.MOCK,
    tokens: List[str] = None,
    event_bus_callback: Optional[Callable] = None,
) -> OnChainFeed:
    """获取或创建 OnChainFeed 单例"""
    global _onchain_feed_instance
    if _onchain_feed_instance is None:
        _onchain_feed_instance = OnChainFeed(
            source=source,
            tokens=tokens,
            event_bus_callback=event_bus_callback,
        )
    return _onchain_feed_instance


def reset_onchain_feed():
    """重置单例（测试用）"""
    global _onchain_feed_instance
    if _onchain_feed_instance:
        _onchain_feed_instance.stop()
    _onchain_feed_instance = None

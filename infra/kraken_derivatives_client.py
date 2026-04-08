"""
Kraken Derivatives REST Client — Feature-gated, fail-closed.

Modes:
  OFF       — completely disabled (default)
  DATA_ONLY — market data ingestion only (funding, OI, mark price)
  LIVE      — full execution (requires KRAKEN_DERIVS_API_KEY/SECRET)

Phase 4: Dynamic instrument discovery via /instruments endpoint.
Phase 5: Derivatives market data fields exposed to cost/risk/signal consumers.
"""

import os
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HMATS.KrakenDerivatives")

PUBLIC_BASE = "https://futures.kraken.com/derivatives/api/v3"

# Static fallback map — used ONLY when /instruments query fails
_FALLBACK_PERP_MAP = {
    "BTC": "PF_XBTUSD",
    "ETH": "PF_ETHUSD",
    "SOL": "PF_SOLUSD",
}

# Assets we care about (logical names)
TARGET_ASSETS = {"BTC": "XBT", "ETH": "ETH", "SOL": "SOL"}


class DerivativesMode(Enum):
    OFF = "off"
    DATA_ONLY = "data_only"
    LIVE = "live"


@dataclass
class InstrumentInfo:
    """Resolved perpetual instrument metadata."""
    symbol: str = ""          # e.g. PF_XBTUSD
    asset: str = ""           # e.g. BTC
    instrument_type: str = "" # e.g. flexible_futures
    tradeable: bool = False
    tick_size: float = 0.0
    contract_size: float = 1.0
    max_leverage: float = 50.0
    margin_levels: List[Dict] = field(default_factory=list)


@dataclass
class DerivativesMarketData:
    """Derivatives market data fields for cost/risk/signal consumers."""
    funding_rate: float = 0.0
    funding_rate_prediction: float = 0.0
    open_interest_usd: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0
    last_price: float = 0.0
    premium: float = 0.0
    volume_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_bps: float = 0.0
    timestamp: float = 0.0


@dataclass
class DerivativesConfig:
    """Configuration for Kraken Derivatives access."""
    mode: DerivativesMode = DerivativesMode.OFF
    api_key: str = ""
    api_secret: str = ""
    max_leverage: float = 5.0
    rate_limit_tokens: int = 500
    rate_limit_refill_per_sec: int = 50

    @classmethod
    def from_env(cls) -> "DerivativesConfig":
        mode_str = os.environ.get("KRAKEN_DERIVS_MODE", "off").lower().strip()
        try:
            mode = DerivativesMode(mode_str)
        except ValueError:
            mode = DerivativesMode.OFF
        return cls(
            mode=mode,
            api_key=os.environ.get("KRAKEN_DERIVS_API_KEY", ""),
            api_secret=os.environ.get("KRAKEN_DERIVS_API_SECRET", ""),
            max_leverage=float(os.environ.get("KRAKEN_DERIVS_MAX_LEVERAGE", "5.0")),
        )


def _http_get_json(url: str, params: Optional[Dict] = None, timeout: int = 10) -> Optional[Dict]:
    """Simple synchronous HTTP GET returning JSON. No external deps beyond stdlib."""
    try:
        import urllib.request
        import urllib.parse
        import json as _json
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "HMATS/6.8"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read())
    except Exception as e:
        logger.debug(f"[DERIVS] HTTP GET {url} failed: {e}")
        return None


class KrakenDerivativesClient:
    """Kraken Derivatives REST client with feature-gated access levels."""

    def __init__(self, config: Optional[DerivativesConfig] = None):
        self.config = config or DerivativesConfig.from_env()
        self.mode = self.config.mode
        self._instruments: Dict[str, InstrumentInfo] = {}
        self._last_market_data: Dict[str, DerivativesMarketData] = {}
        self._last_fetch_ts: float = 0.0
        self._discovery_complete: bool = False

        if self.mode == DerivativesMode.OFF:
            logger.info("[DERIVS] Mode=OFF — derivatives disabled")
            return

        if self.mode == DerivativesMode.LIVE and not self.config.api_key:
            logger.warning("[DERIVS] Mode=LIVE but no API key — falling back to DATA_ONLY")
            self.mode = DerivativesMode.DATA_ONLY

        logger.info(f"[DERIVS] Mode={self.mode.value}")
        self._discover_instruments()

    # ------------------------------------------------------------------
    # Phase 4: Dynamic instrument discovery
    # ------------------------------------------------------------------
    def _discover_instruments(self):
        """Query /instruments endpoint to resolve current USD perpetuals.

        Falls back to static map if API is unreachable.
        """
        data = _http_get_json(f"{PUBLIC_BASE}/instruments")
        if data and "instruments" in data:
            found = 0
            for inst in data["instruments"]:
                symbol = inst.get("symbol", "")
                itype = inst.get("type", "")
                tradeable = bool(inst.get("tradeable", False))

                # Match PF_<ASSET>USD perpetuals for our target assets
                for asset, exchange_name in TARGET_ASSETS.items():
                    expected = f"PF_{exchange_name}USD"
                    if symbol == expected and tradeable:
                        self._instruments[asset] = InstrumentInfo(
                            symbol=symbol,
                            asset=asset,
                            instrument_type=itype,
                            tradeable=tradeable,
                            tick_size=float(inst.get("tickSize", 0) or 0),
                            contract_size=float(inst.get("contractSize", 1) or 1),
                            max_leverage=float(inst.get("maxPositionSize", 50) or 50),
                            margin_levels=inst.get("marginLevels", []),
                        )
                        found += 1
                        break

            self._discovery_complete = found > 0
            if found > 0:
                logger.info(
                    f"[DERIVS] Dynamic discovery: {found}/3 instruments resolved — "
                    + ", ".join(f"{a}={i.symbol}(tick={i.tick_size})" for a, i in self._instruments.items())
                )
            else:
                logger.warning("[DERIVS] Dynamic discovery returned 0 matching instruments — using fallback")
                self._apply_fallback()
        else:
            logger.warning("[DERIVS] /instruments query failed — using static fallback map")
            self._apply_fallback()

        # Log any missing assets
        for asset in TARGET_ASSETS:
            if asset not in self._instruments:
                logger.warning(f"[DERIVS] {asset}: no accessible perpetual found")

    def _apply_fallback(self):
        """Apply static fallback symbol map."""
        for asset, symbol in _FALLBACK_PERP_MAP.items():
            if asset not in self._instruments:
                self._instruments[asset] = InstrumentInfo(
                    symbol=symbol, asset=asset, instrument_type="flexible_futures",
                    tradeable=True, tick_size=0.01, contract_size=1.0,
                )
        self._discovery_complete = False

    def get_instrument(self, asset: str) -> Optional[InstrumentInfo]:
        """Get resolved instrument for an asset. Returns None if unavailable."""
        return self._instruments.get(asset.upper())

    def get_perp_symbol(self, asset: str) -> Optional[str]:
        """Get perpetual symbol string for an asset."""
        inst = self._instruments.get(asset.upper())
        return inst.symbol if inst else None

    def is_discovery_complete(self) -> bool:
        """True if instruments were resolved from live API (not fallback)."""
        return self._discovery_complete

    # ------------------------------------------------------------------
    # Phase 5: Derivatives market data
    # ------------------------------------------------------------------
    def fetch_market_data(self) -> Dict[str, DerivativesMarketData]:
        """Fetch latest derivatives market data for all discovered instruments.

        Returns dict keyed by asset with DerivativesMarketData objects.
        Usable by cost model, risk engine, and signal context.
        """
        if self.mode == DerivativesMode.OFF:
            return {}

        now = time.time()
        if now - self._last_fetch_ts < 30.0 and self._last_market_data:
            return self._last_market_data

        data = _http_get_json(f"{PUBLIC_BASE}/tickers")
        if not data:
            return self._last_market_data

        reverse = {i.symbol: a for a, i in self._instruments.items()}
        result: Dict[str, DerivativesMarketData] = {}

        for ticker in data.get("tickers", []):
            symbol = ticker.get("symbol", "")
            asset = reverse.get(symbol)
            if not asset:
                continue

            mark = float(ticker.get("markPrice", 0) or 0)
            bid = float(ticker.get("bid", 0) or 0)
            ask = float(ticker.get("ask", 0) or 0)
            spread_bps = ((ask - bid) / mark * 10000) if mark > 0 and ask > bid else 0.0

            result[asset] = DerivativesMarketData(
                funding_rate=float(ticker.get("fundingRate", 0) or 0),
                funding_rate_prediction=float(ticker.get("fundingRatePrediction", 0) or 0),
                open_interest_usd=float(ticker.get("openInterest", 0) or 0) * mark if mark > 0 else 0.0,
                mark_price=mark,
                index_price=float(ticker.get("indexPrice", 0) or 0),
                last_price=float(ticker.get("last", 0) or 0),
                premium=float(ticker.get("premium", 0) or 0),
                volume_24h=float(ticker.get("vol24h", 0) or 0),
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                timestamp=now,
            )

        self._last_market_data = result
        self._last_fetch_ts = now
        return result

    def get_funding_history(self, asset: str, count: int = 100) -> List[Dict]:
        """Fetch historical funding rates for a perpetual."""
        if self.mode == DerivativesMode.OFF:
            return []
        symbol = self.get_perp_symbol(asset)
        if not symbol:
            return []
        data = _http_get_json(f"{PUBLIC_BASE}/historicalfundingrates", {"symbol": symbol})
        if not data:
            return []
        return [
            {
                "timestamp": r.get("effectiveTime", ""),
                "funding_rate": float(r.get("fundingRate", 0) or 0),
                "relative_funding_rate": float(r.get("relativeFundingRate", 0) or 0),
            }
            for r in data.get("rates", [])[:count]
        ]

    def is_execution_enabled(self) -> bool:
        return self.mode == DerivativesMode.LIVE and bool(self.config.api_key)

    def get_margin_state(self) -> Optional[Dict[str, Any]]:
        """Get account margin state (LIVE mode only). Returns None if unavailable."""
        if not self.is_execution_enabled():
            return None
        # Requires authenticated API — not implemented until LIVE mode is activated
        return None


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_client: Optional[KrakenDerivativesClient] = None


def get_derivatives_client() -> Optional[KrakenDerivativesClient]:
    """Get or create derivatives client singleton."""
    global _client
    if _client is None:
        config = DerivativesConfig.from_env()
        if config.mode == DerivativesMode.OFF:
            return None
        _client = KrakenDerivativesClient(config)
    return _client


def reset_derivatives_client():
    """Reset singleton (for testing)."""
    global _client
    _client = None

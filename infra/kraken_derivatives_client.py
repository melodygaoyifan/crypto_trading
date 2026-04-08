"""
Kraken Derivatives REST Client — Feature-gated, fail-closed.

Modes:
  OFF       — completely disabled (default)
  DATA_ONLY — market data ingestion only (funding, OI, mark price)
  LIVE      — full execution (requires KRAKEN_DERIVS_API_KEY/SECRET)

Usage:
    from infra.kraken_derivatives_client import get_derivatives_client, DerivativesMode

    client = get_derivatives_client()
    if client and client.mode != DerivativesMode.OFF:
        tickers = client.get_tickers()
"""

import os
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HMATS.KrakenDerivatives")

# Public API base
PUBLIC_BASE = "https://futures.kraken.com/derivatives/api/v3"

# Perpetual symbol map (PF_ = multi-collateral perpetual)
PERP_SYMBOL_MAP = {
    "BTC": "PF_XBTUSD",
    "ETH": "PF_ETHUSD",
    "SOL": "PF_SOLUSD",
}
REVERSE_PERP_MAP = {v: k for k, v in PERP_SYMBOL_MAP.items()}


class DerivativesMode(Enum):
    OFF = "off"
    DATA_ONLY = "data_only"
    LIVE = "live"


@dataclass
class DerivativesConfig:
    """Configuration for Kraken Derivatives access."""
    mode: DerivativesMode = DerivativesMode.OFF
    api_key: str = ""
    api_secret: str = ""
    max_leverage: float = 5.0
    # Rate limit budget (Kraken Futures: 500 tokens, refill 500/10s)
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


class KrakenDerivativesClient:
    """Kraken Derivatives REST client with feature-gated access levels.

    - OFF: all methods return None / empty
    - DATA_ONLY: public market data only (no auth needed)
    - LIVE: full execution (requires API key)
    """

    def __init__(self, config: Optional[DerivativesConfig] = None):
        self.config = config or DerivativesConfig.from_env()
        self.mode = self.config.mode
        self._session = None
        self._last_tickers: Dict[str, Dict] = {}
        self._last_fetch_ts: float = 0.0
        self._instruments: Dict[str, str] = {}  # asset -> perp symbol

        if self.mode == DerivativesMode.OFF:
            logger.info("[DERIVS] Mode=OFF — derivatives disabled")
            return

        if self.mode == DerivativesMode.LIVE and not self.config.api_key:
            logger.warning("[DERIVS] Mode=LIVE but no API key — falling back to DATA_ONLY")
            self.mode = DerivativesMode.DATA_ONLY

        logger.info(f"[DERIVS] Mode={self.mode.value}")
        self._discover_instruments()

    def _discover_instruments(self):
        """Discover available perpetual instruments for BTC/ETH/SOL."""
        for asset, symbol in PERP_SYMBOL_MAP.items():
            self._instruments[asset] = symbol
        logger.info(
            f"[DERIVS] Instruments: {', '.join(f'{a}={s}' for a, s in self._instruments.items())}"
        )

    def get_perp_symbol(self, asset: str) -> Optional[str]:
        """Get perpetual symbol for an asset. Returns None if not available."""
        return self._instruments.get(asset.upper())

    def get_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest ticker data for all perpetuals.

        Returns dict keyed by asset (BTC/ETH/SOL) with funding, OI, mark price.
        Available in DATA_ONLY and LIVE modes.
        """
        if self.mode == DerivativesMode.OFF:
            return {}

        now = time.time()
        if now - self._last_fetch_ts < 30.0 and self._last_tickers:
            return self._last_tickers

        try:
            import requests
            resp = requests.get(f"{PUBLIC_BASE}/tickers", timeout=10)
            resp.raise_for_status()
            data = resp.json()

            result = {}
            for ticker in data.get("tickers", []):
                symbol = ticker.get("symbol", "")
                asset = REVERSE_PERP_MAP.get(symbol)
                if not asset:
                    continue
                result[asset] = {
                    "symbol": symbol,
                    "funding_rate": float(ticker.get("fundingRate", 0) or 0),
                    "funding_rate_prediction": float(ticker.get("fundingRatePrediction", 0) or 0),
                    "open_interest": float(ticker.get("openInterest", 0) or 0),
                    "mark_price": float(ticker.get("markPrice", 0) or 0),
                    "index_price": float(ticker.get("indexPrice", 0) or 0),
                    "last_price": float(ticker.get("last", 0) or 0),
                    "volume_24h": float(ticker.get("vol24h", 0) or 0),
                    "bid": float(ticker.get("bid", 0) or 0),
                    "ask": float(ticker.get("ask", 0) or 0),
                    "premium": float(ticker.get("premium", 0) or 0),
                    "timestamp": now,
                }

            self._last_tickers = result
            self._last_fetch_ts = now
            return result

        except Exception as e:
            logger.warning(f"[DERIVS] Ticker fetch failed: {e}")
            return self._last_tickers

    def get_funding_history(self, asset: str, count: int = 100) -> List[Dict]:
        """Fetch historical funding rates for a perpetual.

        Available in DATA_ONLY and LIVE modes.
        """
        if self.mode == DerivativesMode.OFF:
            return []

        symbol = self.get_perp_symbol(asset)
        if not symbol:
            return []

        try:
            import requests
            resp = requests.get(
                f"{PUBLIC_BASE}/historicalfundingrates",
                params={"symbol": symbol},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            rates = data.get("rates", [])[:count]
            return [
                {
                    "timestamp": r.get("effectiveTime", ""),
                    "funding_rate": float(r.get("fundingRate", 0) or 0),
                    "relative_funding_rate": float(r.get("relativeFundingRate", 0) or 0),
                }
                for r in rates
            ]
        except Exception as e:
            logger.warning(f"[DERIVS] Funding history fetch failed for {asset}: {e}")
            return []

    def is_execution_enabled(self) -> bool:
        """Check if derivatives execution is available."""
        return self.mode == DerivativesMode.LIVE and bool(self.config.api_key)

    def get_margin_state(self) -> Optional[Dict[str, Any]]:
        """Get account margin state (LIVE mode only).

        Returns None if not in LIVE mode or fetch fails.
        """
        if not self.is_execution_enabled():
            return None
        # Placeholder — requires authenticated API call
        logger.debug("[DERIVS] Margin state: requires authenticated implementation")
        return None


# Singleton
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

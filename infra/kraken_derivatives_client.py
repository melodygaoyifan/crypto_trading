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
import base64
import hashlib
import hmac
import urllib.parse
import urllib.request
import urllib.error
import json as _json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HMATS.KrakenDerivatives")

PUBLIC_BASE = "https://futures.kraken.com/derivatives/api/v3"
# Path prefix included in Authent signing string (per Kraken Futures REST spec).
# Full URL is https://futures.kraken.com{_SIGN_PATH_PREFIX}/<endpoint>.
_SIGN_PATH_PREFIX = "/api/v3"

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
            # [P50 2026-04-25] Default lowered 5.0 → 3.0 to align with spot
            # MAX_LEVERAGE (configs/canonical_config.py:95). Operator can still
            # override via env var, but the silent class-default no longer
            # exceeds the spot-side cap on accidental activation.
            max_leverage=float(os.environ.get("KRAKEN_DERIVS_MAX_LEVERAGE", "3.0")),
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

    # ------------------------------------------------------------------
    # Authentication (Kraken Futures REST spec)
    # ------------------------------------------------------------------
    # Signing algorithm (per docs.kraken.com/api/docs/guides/futures-rest/):
    #   1. sha256_digest = SHA256(postData + nonce + endpointPath)
    #   2. secret_bytes  = base64_decode(api_secret)
    #   3. hmac_digest   = HMAC-SHA512(secret_bytes, sha256_digest)
    #   4. Authent       = base64_encode(hmac_digest)
    # Headers sent: APIKey, Authent, Nonce (optional but recommended).
    # ------------------------------------------------------------------
    def _sign(self, endpoint_path: str, post_data: str, nonce: str) -> str:
        """Generate Authent header for a Kraken Futures private endpoint.

        endpoint_path is the path AFTER /derivatives (e.g. "/api/v3/sendorder"),
        matching _SIGN_PATH_PREFIX + the endpoint name.
        """
        if not self.config.api_secret:
            raise RuntimeError("API secret not configured")
        message = (post_data + nonce + endpoint_path).encode("utf-8")
        sha256_digest = hashlib.sha256(message).digest()
        try:
            secret_bytes = base64.b64decode(self.config.api_secret)
        except Exception as e:
            raise RuntimeError(f"API secret is not valid base64: {e}")
        hmac_digest = hmac.new(secret_bytes, sha256_digest, hashlib.sha512).digest()
        return base64.b64encode(hmac_digest).decode("utf-8")

    def _auth_headers(self, endpoint_path: str, post_data: str) -> Dict[str, str]:
        nonce = str(int(time.time() * 1000))
        authent = self._sign(endpoint_path, post_data, nonce)
        return {
            "APIKey": self.config.api_key,
            "Authent": authent,
            "Nonce": nonce,
            "User-Agent": "HMATS/6.8",
        }

    def _private_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 15,
    ) -> Optional[Dict[str, Any]]:
        """Authenticated request helper.

        endpoint: e.g. "sendorder" (without leading /api/v3/ prefix)
        method:   "GET" or "POST"
        params:   dict of params. For POST they go in body; for GET in query string.
        """
        if not self.is_execution_enabled():
            logger.warning(f"[DERIVS] Private call {endpoint} refused: execution not enabled")
            return None

        endpoint_path = f"{_SIGN_PATH_PREFIX}/{endpoint}"
        params = params or {}
        post_data = urllib.parse.urlencode(sorted(params.items()))

        if method.upper() == "GET":
            url = f"{PUBLIC_BASE}/{endpoint}"
            if post_data:
                url = f"{url}?{post_data}"
            data_bytes = None
        else:
            url = f"{PUBLIC_BASE}/{endpoint}"
            data_bytes = post_data.encode("utf-8") if post_data else b""

        headers = self._auth_headers(endpoint_path, post_data)
        if method.upper() == "POST":
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return _json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:400]
            logger.error(f"[DERIVS] HTTP {e.code} on {endpoint}: {body}")
            return {"result": "error", "http_status": e.code, "body": body}
        except Exception as e:
            logger.error(f"[DERIVS] {endpoint} failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Order execution (Phase 6: LIVE execution)
    # ------------------------------------------------------------------
    def send_order(
        self,
        symbol: str,
        side: str,                 # "buy" or "sell"
        size: float,               # contract size (NOT USD — already converted by caller)
        order_type: str = "lmt",   # "lmt" | "mkt" | "post" | "stp" | "take_profit" | "ioc"
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        cli_ord_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Place an order on Kraken Futures.

        POST /derivatives/api/v3/sendorder

        Returns the parsed JSON response. On success, response has
        result="success" and contains sendStatus with order_id + status.
        """
        params: Dict[str, Any] = {
            "orderType": order_type,
            "symbol": symbol,
            "side": side,
            "size": f"{size:.8f}".rstrip("0").rstrip("."),
        }
        if limit_price is not None:
            params["limitPrice"] = f"{limit_price:.8f}".rstrip("0").rstrip(".")
        if stop_price is not None:
            params["stopPrice"] = f"{stop_price:.8f}".rstrip("0").rstrip(".")
        if reduce_only:
            params["reduceOnly"] = "true"
        if cli_ord_id:
            params["cliOrdId"] = cli_ord_id
        return self._private_request("POST", "sendorder", params)

    def cancel_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """POST /derivatives/api/v3/cancelorder"""
        return self._private_request("POST", "cancelorder", {"order_id": order_id})

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """GET /derivatives/api/v3/openpositions — used for reconciliation.

        Returns list of position dicts with: symbol, side, size, price, fillTime,
        unrealizedFunding, pnlCurrency.
        """
        resp = self._private_request("GET", "openpositions")
        if not resp or resp.get("result") != "success":
            return []
        return resp.get("openPositions", []) or []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """GET /derivatives/api/v3/openorders"""
        resp = self._private_request("GET", "openorders")
        if not resp or resp.get("result") != "success":
            return []
        return resp.get("openOrders", []) or []

    def get_accounts(self) -> Optional[Dict[str, Any]]:
        """GET /derivatives/api/v3/accounts — margin + balance state."""
        resp = self._private_request("GET", "accounts")
        if not resp or resp.get("result") != "success":
            return None
        return resp.get("accounts")

    def get_margin_state(self) -> Optional[Dict[str, Any]]:
        """Get account margin state (LIVE mode only). Returns None if unavailable."""
        return self.get_accounts()

    # ------------------------------------------------------------------
    # Mark price helper (convenience for DerivativesExecutor)
    # ------------------------------------------------------------------
    def get_mark_price(self, asset_or_symbol: str) -> Optional[float]:
        """Return latest cached mark price for an asset (e.g. "BTC") or symbol
        (e.g. "PF_XBTUSD"). Triggers a fetch if cache is stale."""
        # Resolve asset from symbol if needed
        if asset_or_symbol in self._instruments:
            asset = asset_or_symbol
        else:
            asset = next(
                (a for a, i in self._instruments.items() if i.symbol == asset_or_symbol),
                None,
            )
        if asset is None:
            return None
        data = self.fetch_market_data()
        md = data.get(asset)
        return md.mark_price if md and md.mark_price > 0 else None


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

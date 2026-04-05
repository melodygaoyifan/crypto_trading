"""
================================================================================
HMATS v6 - MARKET DATA SCHEMA (Legacy/Compat)
================================================================================
Purpose: Validate and bridge market data between legacy nested format and v6 flat
contract.  Retained for backward compatibility - v6 runtime may pass flat dicts
directly.  This module normalises both formats and provides staleness / alias logic.

RULE: If required keys are missing -> NO_TRADE (fail-safe)
RULE: Missing safety fields stay None (fail-closed), never forced to "safe" values.

Schema:
  market_data = {
      "BTC": OHLCVData,     # Required
      "ETH": OHLCVData,     # Required
      "SOL": OHLCVData,     # Required
      "timestamp": datetime,
      "data_age_seconds": float,

      # Optional microstructure (top-level, None = unknown)
      "vpin": Optional[float],
      "ofi": Optional[float],
      "dvol_zscore": Optional[float],
      "correlation_btc_sol": float,
      "correlation_eth_sol": float,
      "correlation_btc_eth_sol": Optional[float],

      # Optional v6 execution context
      "orderbook_depth": Optional[float],
      "orderbook_depth_1pct_usd": Optional[float],
      "estimated_friction_bps": Optional[float],
      "spread_bps": Optional[float],
      "order_book_imbalance": Optional[float],
      "flash_crash_active": Optional[bool],
      "price_move_pct": Optional[float],
  }

  OHLCVData = {
      "open": float,
      "high": float,
      "low": float,
      "close": float,
      "volume": float,
      "returns_4h": float,      # Computed: (close - prev_close) / prev_close
      "volatility_4h": float,   # Computed: rolling std of returns

      # Optional OHLCV history for indicator calculation
      "ohlcv_history": pd.DataFrame,  # 200+ bars for EMA calculation
  }
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime, timezone
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class MarketDataConfig:
    """Configuration for market data validation."""

    # Required assets
    REQUIRED_ASSETS = ["BTC", "ETH", "SOL"]

    # Required OHLCV fields per asset
    REQUIRED_OHLCV_FIELDS = ["open", "high", "low", "close", "volume"]

    # Computed fields (generated if missing)
    COMPUTED_FIELDS = ["returns_4h", "volatility_4h"]

    # Optional microstructure fields
    OPTIONAL_MICRO_FIELDS = [
        "vpin", "ofi", "dvol_zscore",
        "correlation_btc_sol", "correlation_eth_sol", "correlation_btc_eth_sol",
    ]

    # Optional v6 execution context fields
    OPTIONAL_V6_FIELDS = [
        "orderbook_depth", "orderbook_depth_1pct_usd",
        "estimated_friction_bps", "spread_bps",
        "order_book_imbalance", "flash_crash_active", "price_move_pct",
    ]

    # Dual staleness thresholds (seconds)
    # OHLCV is slow-moving (4H bars) - 10s matches main.py MAX_DATA_AGE_SECONDS
    MAX_OHLCV_AGE_SECONDS = 10.0
    # Microstructure decays fast - 2s for vpin/ofi/orderbook
    MAX_MICRO_AGE_SECONDS = 2.0

    # Backward compat alias
    MAX_DATA_AGE_SECONDS = MAX_OHLCV_AGE_SECONDS

    # Minimum history bars for indicator calculation
    MIN_HISTORY_BARS = 200

    # Alias mapping: incoming raw key -> canonical field name
    FIELD_ALIASES: Dict[str, str] = {
        "dvol": "dvol_zscore",
        "price": "current_price",
        "correlation": "correlation_btc_sol",
    }


# =============================================================================
# OHLCV DATA CLASS
# =============================================================================

@dataclass
class OHLCVData:
    """Canonical OHLCV data for a single asset."""
    
    # Core OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Computed
    returns_4h: float = 0.0
    volatility_4h: float = 0.02
    
    # Optional history for indicator calculation
    ohlcv_history: Optional[Any] = None  # pd.DataFrame
    
    def __post_init__(self):
        """Validate on creation."""
        self._validate()
    
    def _validate(self):
        """Validate OHLCV values."""
        if self.close <= 0:
            raise ValueError(f"Invalid close price: {self.close}")
        if self.volume < 0:
            raise ValueError(f"Invalid volume: {self.volume}")
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) < Low ({self.low})")
    
    def to_dict(self) -> Dict:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "returns_4h": self.returns_4h,
            "volatility_4h": self.volatility_4h,
            "has_history": self.ohlcv_history is not None
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "OHLCVData":
        """Create from dictionary."""
        return cls(
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            volume=d["volume"],
            returns_4h=d.get("returns_4h", 0.0),
            volatility_4h=d.get("volatility_4h", 0.02),
            ohlcv_history=d.get("ohlcv_history")
        )


# =============================================================================
# CANONICAL MARKET DATA
# =============================================================================

@dataclass
class CanonicalMarketData:
    """
    The ONLY acceptable market data structure for HMATS v6.

    All pipeline components MUST use this schema.
    Fail-closed: missing safety fields stay None, never forced to "safe" values.
    """

    # Per-asset OHLCV
    BTC: OHLCVData
    ETH: OHLCVData
    SOL: OHLCVData

    # Timestamp
    timestamp: datetime
    data_age_seconds: float = 0.0

    # Optional microstructure (None = unknown, fail-closed)
    vpin: Optional[float] = None
    ofi: Optional[float] = None
    dvol_zscore: Optional[float] = None
    correlation_btc_sol: float = 0.8
    correlation_eth_sol: float = 0.7
    correlation_btc_eth_sol: Optional[float] = None

    # Optional v6 execution context (None = unknown)
    orderbook_depth: Optional[float] = None
    orderbook_depth_1pct_usd: Optional[float] = None
    estimated_friction_bps: Optional[float] = None
    spread_bps: Optional[float] = None
    order_book_imbalance: Optional[float] = None
    flash_crash_active: Optional[bool] = None
    price_move_pct: Optional[float] = None

    # Validation state
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def get_asset(self, asset: str) -> OHLCVData:
        """Get OHLCV data for an asset."""
        if asset == "BTC":
            return self.BTC
        elif asset == "ETH":
            return self.ETH
        elif asset == "SOL":
            return self.SOL
        else:
            raise ValueError(f"Unknown asset: {asset}")
    
    def to_legacy_dict(self, primary_asset: str = "SOL") -> Dict:
        """
        Convert to legacy flat dict format for backward compatibility.

        Args:
            primary_asset: Asset whose price populates "current_price", "volume",
                "returns_4h", and "volatility_4h" top-level keys.
                Defaults to "SOL" for backward compatibility.

        This bridges the canonical schema with existing code that expects:
        market_data = {"btc_price": ..., "current_price": ...}
        """
        primary = self.get_asset(primary_asset)
        d = {
            # Legacy keys - primary_asset drives "current_price"
            "current_price": primary.close,
            "btc_price": self.BTC.close,
            "eth_price": self.ETH.close,
            "sol_price": self.SOL.close,

            # Volume
            "volume": primary.volume,
            "btc_volume": self.BTC.volume,
            "eth_volume": self.ETH.volume,

            # Returns
            "returns_4h": primary.returns_4h,
            "btc_returns_4h": self.BTC.returns_4h,
            "eth_returns_4h": self.ETH.returns_4h,

            # Volatility
            "volatility_4h": primary.volatility_4h,

            # Microstructure (None = unknown)
            "vpin": self.vpin,
            "ofi": self.ofi,
            "dvol_zscore": self.dvol_zscore,
            "correlation": self.correlation_btc_sol,
            "correlation_btc_sol": self.correlation_btc_sol,
            "correlation_btc_eth_sol": self.correlation_btc_eth_sol,

            # Metadata
            "timestamp": self.timestamp,
            "data_age_seconds": self.data_age_seconds,

            # Per-asset dict for new code
            "BTC": self.BTC.to_dict(),
            "ETH": self.ETH.to_dict(),
            "SOL": self.SOL.to_dict(),
        }

        # V6 execution context - only include non-None fields
        for fld in ("orderbook_depth", "orderbook_depth_1pct_usd",
                     "estimated_friction_bps", "spread_bps",
                     "order_book_imbalance", "flash_crash_active",
                     "price_move_pct"):
            val = getattr(self, fld, None)
            if val is not None:
                d[fld] = val

        return d

    def to_asset_market_data(self, asset: str) -> Dict[str, Any]:
        """
        Return the per-asset market_data dict expected by v6 agents.

        Includes the 4 execution-critical keys (numeric/bool, never None):
          orderbook_depth, estimated_friction_bps, flash_crash_active,
          price_move_pct - all default to safe values when unavailable.

        Also provides alias keys (dvol -> dvol_zscore, order_book_imbalance).
        """
        ohlcv = self.get_asset(asset)
        al = asset.lower()

        # Compute price_move_pct from OHLCV if not already set
        price_move = self.price_move_pct
        if price_move is None and ohlcv.open > 0:
            price_move = (ohlcv.close - ohlcv.open) / ohlcv.open * 100.0

        # Compute estimated_friction_bps from spread if not already set
        friction = self.estimated_friction_bps
        if friction is None and self.spread_bps is not None:
            friction = self.spread_bps / 2.0

        # Flash crash heuristic: |price_move| > 8% in one bar
        flash = self.flash_crash_active
        if flash is None:
            flash = abs(price_move or 0.0) > 8.0

        d: Dict[str, Any] = {
            # Per-asset OHLCV
            "current_price": ohlcv.close,
            f"{al}_price": ohlcv.close,
            "open": ohlcv.open,
            "high": ohlcv.high,
            "low": ohlcv.low,
            "close": ohlcv.close,
            "volume": ohlcv.volume,
            "returns_4h": ohlcv.returns_4h,
            "volatility_4h": ohlcv.volatility_4h,

            # Cross-asset prices (always present)
            "btc_price": self.BTC.close,
            "eth_price": self.ETH.close,
            "sol_price": self.SOL.close,

            # Microstructure (None = unknown)
            "vpin": self.vpin,
            "ofi": self.ofi,
            "dvol_zscore": self.dvol_zscore,
            "dvol": self.dvol_zscore,  # alias
            "correlation_btc_sol": self.correlation_btc_sol,
            "correlation_eth_sol": self.correlation_eth_sol,
            "correlation_btc_eth_sol": self.correlation_btc_eth_sol,
            "correlation": self.correlation_btc_sol,  # alias

            # 4 execution-critical keys (always numeric/bool, never None)
            "orderbook_depth": self.orderbook_depth if self.orderbook_depth is not None else 0.0,
            "estimated_friction_bps": friction if friction is not None else 0.0,
            "flash_crash_active": flash,
            "price_move_pct": price_move if price_move is not None else 0.0,

            # Additional execution context
            "spread_bps": self.spread_bps,
            "order_book_imbalance": self.order_book_imbalance,
            "orderbook_depth_1pct_usd": self.orderbook_depth_1pct_usd,

            # Metadata
            "timestamp": self.timestamp,
            "data_age_seconds": self.data_age_seconds,
            "asset": asset,
        }

        return d


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

class MarketDataValidator:
    """
    Validate and normalize market data.
    
    RULE: If validation fails -> return (None, error) -> triggers NO_TRADE
    """
    
    def __init__(self):
        self.config = MarketDataConfig()
    
    def validate(
        self,
        raw_data: Dict,
        primary_asset: str = "SOL",
    ) -> Tuple[Optional[CanonicalMarketData], str]:
        """
        Validate raw market data and convert to canonical form.

        Args:
            raw_data: Raw market data dict (any format).
            primary_asset: Asset whose price may be in ``current_price``
                (backward-compat). Defaults to SOL only for legacy callers.

        Returns:
            (CanonicalMarketData, "") if valid
            (None, error_message) if invalid -> triggers NO_TRADE
        """
        errors = []

        # Apply field aliases (dvol->dvol_zscore, price->current_price, etc.)
        aliased = dict(raw_data)
        for alias, canonical_key in self.config.FIELD_ALIASES.items():
            if alias in aliased and canonical_key not in aliased:
                aliased[canonical_key] = aliased[alias]

        # Check timestamp
        timestamp = aliased.get("timestamp", datetime.now(timezone.utc))
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except Exception:
                timestamp = datetime.now(timezone.utc)

        # Dual staleness check
        data_age = aliased.get("data_age_seconds", 0.0)
        micro_age = aliased.get("micro_data_age_seconds", data_age)
        if data_age > self.config.MAX_OHLCV_AGE_SECONDS:
            return None, (f"OHLCV stale: {data_age:.1f}s "
                          f"> {self.config.MAX_OHLCV_AGE_SECONDS}s")

        # Validate each required asset
        asset_data = {}
        for asset in self.config.REQUIRED_ASSETS:
            ohlcv, error = self._extract_asset_ohlcv(
                aliased, asset, primary_asset=primary_asset,
            )
            if error:
                errors.append(f"{asset}: {error}")
            else:
                asset_data[asset] = ohlcv

        # If any required asset missing -> fail
        if len(asset_data) != len(self.config.REQUIRED_ASSETS):
            return None, f"Missing assets: {', '.join(errors)}"

        # Extract microstructure - fail-closed: None if missing, not "safe" defaults
        micro_stale = micro_age > self.config.MAX_MICRO_AGE_SECONDS
        vpin = None if micro_stale else aliased.get("vpin")
        ofi = None if micro_stale else aliased.get("ofi")
        dvol_zscore = aliased.get("dvol_zscore")
        corr_btc = aliased.get("correlation_btc_sol",
                               aliased.get("correlation", 0.8))
        corr_eth = aliased.get("correlation_eth_sol", 0.7)
        corr_trio = aliased.get("correlation_btc_eth_sol")

        # V6 execution context - None if not provided
        orderbook_depth = aliased.get("orderbook_depth")
        orderbook_depth_1pct = aliased.get("orderbook_depth_1pct_usd")
        friction_bps = aliased.get("estimated_friction_bps")
        spread_bps = aliased.get("spread_bps")
        ob_imbalance = aliased.get("order_book_imbalance")
        flash_crash = aliased.get("flash_crash_active")
        price_move = aliased.get("price_move_pct")

        # Create canonical data
        try:
            canonical = CanonicalMarketData(
                BTC=asset_data["BTC"],
                ETH=asset_data["ETH"],
                SOL=asset_data["SOL"],
                timestamp=timestamp,
                data_age_seconds=data_age,
                vpin=vpin,
                ofi=ofi,
                dvol_zscore=dvol_zscore,
                correlation_btc_sol=corr_btc,
                correlation_eth_sol=corr_eth,
                correlation_btc_eth_sol=corr_trio,
                orderbook_depth=orderbook_depth,
                orderbook_depth_1pct_usd=orderbook_depth_1pct,
                estimated_friction_bps=friction_bps,
                spread_bps=spread_bps,
                order_book_imbalance=ob_imbalance,
                flash_crash_active=flash_crash,
                price_move_pct=price_move,
                is_valid=len(errors) == 0,
                validation_errors=errors
            )

            if errors:
                logger.warning(f"Market data has warnings: {errors}")

            return canonical, ""

        except Exception as e:
            return None, f"Failed to create canonical data: {e}"
    
    def _extract_asset_ohlcv(
        self,
        raw_data: Dict,
        asset: str,
        primary_asset: str = "",
    ) -> Tuple[Optional[OHLCVData], str]:
        """
        Extract OHLCV for a single asset from raw data.

        Handles multiple input formats:
        1. Nested: raw_data["BTC"]["close"]
        2. Flat: raw_data["btc_price"]
        3. Legacy: raw_data["current_price"] (only for primary_asset)
        """

        asset_lower = asset.lower()

        # Try nested format first
        if asset in raw_data and isinstance(raw_data[asset], dict):
            nested = raw_data[asset]
            try:
                return OHLCVData(
                    open=nested.get("open", nested.get("close", 0)),
                    high=nested.get("high", nested.get("close", 0)),
                    low=nested.get("low", nested.get("close", 0)),
                    close=nested["close"],
                    volume=nested.get("volume", 0),
                    returns_4h=nested.get("returns_4h", 0.0),
                    volatility_4h=nested.get("volatility_4h", 0.02),
                    ohlcv_history=nested.get("ohlcv_history")
                ), ""
            except (KeyError, ValueError) as e:
                return None, f"Invalid nested format: {e}"

        # Try flat format: {asset}_price
        price_key = f"{asset_lower}_price"
        close = raw_data.get(price_key, 0)

        # Fallback: "current_price" applies to the PRIMARY asset only (not SOL-specific)
        if close <= 0 and asset == primary_asset:
            close = raw_data.get("current_price", 0)
        
        if close <= 0:
            return None, f"Missing or invalid price"
        
        # Build OHLCV from flat keys
        volume_key = f"{asset_lower}_volume"
        returns_key = f"{asset_lower}_returns_4h"
        
        try:
            return OHLCVData(
                open=raw_data.get(f"{asset_lower}_open", close),
                high=raw_data.get(f"{asset_lower}_high", close * 1.01),
                low=raw_data.get(f"{asset_lower}_low", close * 0.99),
                close=close,
                volume=raw_data.get(volume_key, raw_data.get("volume", 0)),
                returns_4h=raw_data.get(returns_key, raw_data.get("returns_4h", 0.0)),
                volatility_4h=raw_data.get(f"{asset_lower}_volatility_4h", 
                                          raw_data.get("volatility_4h", 0.02)),
                ohlcv_history=raw_data.get(f"{asset_lower}_ohlcv_history")
            ), ""
        except ValueError as e:
            return None, f"Invalid flat format: {e}"


# =============================================================================
# RUNTIME ASSERTION
# =============================================================================

def assert_valid_market_data(raw_data: Dict) -> Tuple[CanonicalMarketData, bool, str]:
    """
    Assert market data is valid. Used at pipeline entry.
    
    Returns:
        (canonical_data, is_valid, error_message)
        
    If is_valid=False -> caller MUST trigger NO_TRADE
    """
    validator = MarketDataValidator()
    canonical, error = validator.validate(raw_data)
    
    if canonical is None:
        logger.error(f"[DATA VALIDATION FAILED] {error} -> NO_TRADE")
        return None, False, error
    
    return canonical, True, ""


# =============================================================================
# V6 FLAT-DICT VALIDATOR
# =============================================================================

# Required keys (must be present *and* non-None for a trade to proceed)
_V6_REQUIRED_KEYS = {"timestamp", "data_age_seconds"}
# Per-asset: at least one price key must exist for each required asset
_V6_PRICE_PATTERNS = ["{asset}_price", "price_{asset}"]
# Optional but recognised - checked for correct type if present
_V6_OPTIONAL_FLOAT_KEYS = {
    "vpin", "ofi", "dvol_zscore",
    "correlation_btc_sol", "correlation_eth_sol", "correlation_btc_eth_sol",
    "orderbook_depth", "orderbook_depth_1pct_usd",
    "estimated_friction_bps", "spread_bps",
    "order_book_imbalance", "price_move_pct",
    "lead_lag_edge", "lead_lag_confidence", "cvd_divergence",
    "signal_edge_bps", "sentiment_zscore",
}
_V6_OPTIONAL_BOOL_KEYS = {"flash_crash_active"}

# Known aliases -> canonical name
_V6_ALIASES: Dict[str, str] = {
    "dvol": "dvol_zscore",
    "price": "current_price",
    "correlation": "correlation_btc_sol",
}


def validate_market_data_v6(
    market_data: Dict[str, Any],
    required_assets: List[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Lightweight v6 flat-dict validator.

    Checks required keys, per-asset price presence, alias resolution, and
    type correctness for optional fields.  Does NOT convert to
    CanonicalMarketData - use MarketDataValidator.validate() for that.

    Args:
        market_data: Flat dict from the v6 runtime.
        required_assets: Assets that must have a price key (default BTC/ETH/SOL).

    Returns:
        (is_valid, list_of_issues)
        is_valid=False  ->  caller should NO_TRADE.
        is_valid=True with issues ->  optional degradation warnings.
    """
    if required_assets is None:
        required_assets = ["BTC", "ETH", "SOL"]

    issues: List[str] = []

    # Apply aliases
    md = dict(market_data)
    for alias, canonical in _V6_ALIASES.items():
        if alias in md and canonical not in md:
            md[canonical] = md[alias]

    # Required keys
    for key in _V6_REQUIRED_KEYS:
        if key not in md:
            issues.append(f"missing required key: {key}")

    # Per-asset price
    for asset in required_assets:
        al = asset.lower()
        has_price = any(
            md.get(pat.format(asset=al)) is not None
            for pat in _V6_PRICE_PATTERNS
        )
        # Also accept nested dict or "current_price" for backward compat
        if not has_price:
            has_price = (
                (asset in md and isinstance(md[asset], dict)
                 and "close" in md[asset])
                or md.get("current_price") is not None
            )
        if not has_price:
            issues.append(f"no price for {asset}")

    # If any required check failed -> invalid
    is_valid = not any(
        i.startswith("missing required") or i.startswith("no price")
        for i in issues
    )

    # Type checks on optional floats (warn, don't fail)
    for key in _V6_OPTIONAL_FLOAT_KEYS:
        val = md.get(key)
        if val is not None and not isinstance(val, (int, float)):
            issues.append(f"{key}: expected float, got {type(val).__name__}")

    for key in _V6_OPTIONAL_BOOL_KEYS:
        val = md.get(key)
        if val is not None and not isinstance(val, bool):
            issues.append(f"{key}: expected bool, got {type(val).__name__}")

    return is_valid, issues


# =============================================================================
# NORMALIZE MARKET DATA (v6 flat-dict transformer)
# =============================================================================

# Execution-critical keys - always present in output (None + diagnostic if absent)
_EXEC_CRITICAL_KEYS = (
    "orderbook_depth", "orderbook_depth_1pct_usd",
    "estimated_friction_bps", "spread_bps",
    "order_book_imbalance", "flash_crash_active", "price_move_pct",
)


def normalize_market_data(
    raw_data: Dict[str, Any],
    primary_asset: str = "SOL",
) -> Dict[str, Any]:
    """
    Transform any raw market data dict into a v6-compatible flat dict.

    Unlike ``validate_market_data_v6`` (which only checks), this function
    *produces* the canonical flat dict that downstream agents expect:

    * ``asset``                 - echoes *primary_asset*
    * ``timestamp``             - from raw or ``utcnow()`` fallback
    * ``data_age_seconds``      - from raw or 0.0
    * ``current_price``         - primary asset close / price
    * ``{asset}_price``         - per-asset prices (BTC, ETH, SOL)
    * execution-critical keys   - always present; ``None`` + diagnostic if absent
    * optional micro / v6 keys  - carried through when present
    * ``_diagnostics``          - ``{"missing": [...], "aliased": [...]}``

    Args:
        raw_data:  Raw market data (nested or flat).
        primary_asset:  Asset whose price populates ``current_price``.

    Returns:
        Flat dict ready for v6 agents.  Never raises.
    """
    out: Dict[str, Any] = {}
    missing: List[str] = []
    aliased: List[str] = []

    # --- 1. shallow copy + alias resolution ---
    md = dict(raw_data)
    for alias, canonical in _V6_ALIASES.items():
        if alias in md and canonical not in md:
            md[canonical] = md[alias]
            aliased.append(f"{alias}->{canonical}")

    # --- 2. primary_asset echo ---
    out["asset"] = primary_asset.upper()

    # --- 3. timestamp ---
    ts = md.get("timestamp")
    if ts is None:
        ts = datetime.now(timezone.utc)
        missing.append("timestamp")
    elif isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except Exception:
            ts = datetime.now(timezone.utc)
            missing.append("timestamp(parse_failed)")
    out["timestamp"] = ts
    out["data_age_seconds"] = md.get("data_age_seconds", 0.0)

    # --- 4. per-asset prices ---
    for asset in ("BTC", "ETH", "SOL"):
        al = asset.lower()
        price_key = f"{al}_price"
        price = md.get(price_key)

        # fallback: nested dict with "close"
        if price is None and asset in md and isinstance(md[asset], dict):
            price = md[asset].get("close")

        # fallback: "current_price" applies to primary asset only
        if price is None and asset == primary_asset.upper():
            price = md.get("current_price")

        if price is not None:
            out[price_key] = float(price)
        else:
            missing.append(price_key)

    # --- 5. current_price = primary asset close ---
    pa = primary_asset.upper()
    pa_price_key = f"{pa.lower()}_price"
    out["current_price"] = out.get(pa_price_key)
    if out["current_price"] is None:
        missing.append("current_price")

    # --- 6. execution-critical keys (always present; None if absent) ---
    for key in _EXEC_CRITICAL_KEYS:
        val = md.get(key)
        out[key] = val
        if val is None:
            missing.append(key)

    # --- 7. optional microstructure / v6 pass-through ---
    for key in _V6_OPTIONAL_FLOAT_KEYS:
        if key not in out and key in md:
            out[key] = md[key]
    for key in _V6_OPTIONAL_BOOL_KEYS:
        if key not in out and key in md:
            out[key] = md[key]

    # --- 8. carry over extra per-asset fields (volume, returns, volatility) ---
    for asset in ("BTC", "ETH", "SOL"):
        al = asset.lower()
        for suffix in ("_volume", "_returns_4h", "_volatility_4h"):
            k = al + suffix
            if k in md:
                out[k] = md[k]
        # nested dict -> extract volume / returns / volatility if not already flat
        if asset in md and isinstance(md[asset], dict):
            nd = md[asset]
            if f"{al}_volume" not in out and "volume" in nd:
                out[f"{al}_volume"] = nd["volume"]
            if f"{al}_returns_4h" not in out and "returns_4h" in nd:
                out[f"{al}_returns_4h"] = nd["returns_4h"]
            if f"{al}_volatility_4h" not in out and "volatility_4h" in nd:
                out[f"{al}_volatility_4h"] = nd["volatility_4h"]

    # top-level volume / returns / volatility (primary asset)
    if "volume" not in out:
        out["volume"] = out.get(f"{pa.lower()}_volume")
    if "returns_4h" not in out:
        out["returns_4h"] = out.get(f"{pa.lower()}_returns_4h")
    if "volatility_4h" not in out:
        out["volatility_4h"] = out.get(f"{pa.lower()}_volatility_4h", md.get("volatility_4h"))

    # --- 9. diagnostics ---
    out["_diagnostics"] = {"missing": missing, "aliased": aliased}

    return out


# =============================================================================
# OHLCV HISTORY BUILDER (for indicator calculation)
# =============================================================================

def build_ohlcv_dataframe(
    history: List[Dict],
    asset: str = "SOL"
) -> "pd.DataFrame":
    """
    Build pandas DataFrame from OHLCV history list.
    
    Args:
        history: List of {"timestamp", "open", "high", "low", "close", "volume"}
        
    Returns:
        pd.DataFrame suitable for technical analysis
    """
    try:
        import pandas as pd
        
        df = pd.DataFrame(history)
        
        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Sort by timestamp if present
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        
        return df
        
    except ImportError:
        logger.warning("pandas not available - cannot build OHLCV DataFrame")
        return None


# =============================================================================
# SINGLETON
# =============================================================================

_validator: Optional[MarketDataValidator] = None


def get_market_data_validator() -> MarketDataValidator:
    """Get singleton validator instance."""
    global _validator
    if _validator is None:
        _validator = MarketDataValidator()
    return _validator


# =============================================================================
# TEST
# =============================================================================

def test_market_data_schema():
    """Smoke test the market data schema validation (run with __main__)."""
    validator = MarketDataValidator()

    # 1. Valid nested format
    nested_data = {
        "BTC": {"open": 95000, "high": 96000, "low": 94000, "close": 95500, "volume": 1000},
        "ETH": {"open": 3400, "high": 3500, "low": 3350, "close": 3450, "volume": 500},
        "SOL": {"open": 180, "high": 190, "low": 175, "close": 185, "volume": 2000,
                "returns_4h": 0.028, "volatility_4h": 0.04},
        "timestamp": datetime.now(timezone.utc),
        "vpin": 0.55,
    }
    canonical, error = validator.validate(nested_data)
    assert canonical is not None, f"Expected valid, got error: {error}"
    assert canonical.vpin == 0.55

    # 2. to_legacy_dict with primary_asset
    leg_sol = canonical.to_legacy_dict(primary_asset="SOL")
    assert leg_sol["current_price"] == 185
    leg_btc = canonical.to_legacy_dict(primary_asset="BTC")
    assert leg_btc["current_price"] == 95500

    # 3. Alias mapping (dvol->dvol_zscore)
    alias_data = dict(nested_data)
    alias_data["dvol"] = -1.5
    canonical2, _ = validator.validate(alias_data)
    assert canonical2.dvol_zscore == -1.5

    # 4. OHLCV stale (>10s) fails
    stale_data = dict(nested_data)
    stale_data["data_age_seconds"] = 15.0
    canon_stale, err = validator.validate(stale_data)
    assert canon_stale is None
    assert "OHLCV stale" in err

    # 5. Micro stale (>2s) nullifies vpin/ofi but passes OHLCV
    micro_stale = dict(nested_data)
    micro_stale["data_age_seconds"] = 5.0
    micro_stale["micro_data_age_seconds"] = 3.0
    micro_stale["vpin"] = 0.99
    canon_ms, _ = validator.validate(micro_stale)
    assert canon_ms is not None
    assert canon_ms.vpin is None  # micro-stale -> None

    # 6. Fail-closed: missing vpin/ofi stay None
    no_micro = dict(nested_data)
    del no_micro["vpin"]
    canon_nm, _ = validator.validate(no_micro)
    assert canon_nm.vpin is None
    assert canon_nm.ofi is None

    print("market_data_schema smoke tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_market_data_schema()

"""
HMATS v5.1 Phase 2.3 - Coinbase Funding-Rate Feed (SCAFFOLD, fail-closed)
=========================================================================

Provides Coinbase perpetual funding rates normalized to an 8h-equivalent,
for the funding-rate strategy layer (v5.1 Phase 3) and for dual-venue
funding parity comparison (Phase 2 SHADOW week).

STATUS: scaffold. Returns None / fail-closed until the access + product
question below is resolved and credentials are wired. Callers MUST treat a
None return as "no Coinbase funding; use the Kraken futures funding feed."

------------------------------------------------------------------------
ACCESS / PRODUCT BLOCKER (verified 2026-06-13, ccxt 4.5.34) — READ FIRST
------------------------------------------------------------------------
Coinbase perps are NOT a single product, and ccxt support is split:

  * ccxt.coinbaseadvanced   swap=False  -> US-accessible, but NO perps via ccxt
  * ccxt.coinbaseinternational swap=True -> perps exist, but the venue is
        US-RESTRICTED (Coinbase International Exchange), AND
        fetchFundingRate=False / fetchFundingRates=False -> funding is only
        reachable via a raw/implicit endpoint, not the ccxt unified method.
  * "US Perpetual-Style Futures" (CFTC-regulated, 5yr-dated, launched 2025)
        are a DIFFERENT product (Coinbase Financial Markets / Derivatives),
        with their own symbols + API and no confirmed ccxt unified support.

So before this feed can be implemented for real, the operator must confirm
WHICH product their account can trade via API (see
docs/COINBASE_MIGRATION_PREP.md). The symbol map currently assumes the
International-style `BTC-PERP`; US Perpetual-Style Futures use different
symbols and a funding mechanism that "accrues hourly, settles twice daily".

This mirrors the v5.1 V13 (Deribit US-restricted) finding: a venue/product
access verification gate, not a coding gap.
------------------------------------------------------------------------

Iron Laws:
  4. fail-closed: every method returns None / raises; callers fall back to
     the Kraken futures funding feed during dual-venue.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from exchange.adapter import FundingRateData
from exchange.symbol_mapping import to_venue_symbol

logger = logging.getLogger(__name__)

# Coinbase perp funding settles intraday; normalize everything to an 8h
# equivalent so it is comparable with the strategy thresholds (which are
# expressed per-8h, matching the v5.1 Phase 3 spec).
_DEFAULT_FUNDING_PERIOD_HOURS = 8.0


class CoinbaseFundingFeed:
    """Scaffold funding feed. Disabled (fail-closed) until product/access
    resolved and credentials wired.

    Args:
        ccxt_client: optional pre-wired ccxt.coinbaseinternational client.
            If None, the feed stays disabled and returns None.
        enabled: master switch; default False so this can be imported and
            constructed with zero live side effects (inert during prep).
    """

    def __init__(self, ccxt_client=None, enabled: bool = False) -> None:
        self._client = ccxt_client
        self._enabled = bool(enabled) and ccxt_client is not None
        self._api_key = os.environ.get("COINBASE_API_KEY")
        self._api_secret = os.environ.get("COINBASE_API_SECRET")
        self._warned_disabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _normalize_to_8h(rate: float, period_hours: float) -> float:
        """Scale a per-period funding rate to its 8h equivalent."""
        if not period_hours or period_hours <= 0:
            return float(rate)
        return float(rate) * (_DEFAULT_FUNDING_PERIOD_HOURS / float(period_hours))

    async def fetch_funding_rate(
        self, asset: str, market: str = "perp"
    ) -> Optional[FundingRateData]:
        """Return Coinbase funding for an HMATS-canonical asset, or None.

        fail-closed: returns None (NOT an exception) when disabled, so the
        caller's `coinbase_funding or kraken_funding` fallback works cleanly.
        """
        if not self._enabled:
            if not self._warned_disabled:
                logger.info(
                    "[COINBASE_FUNDING] disabled (scaffold). Resolve product/"
                    "access per docs/COINBASE_MIGRATION_PREP.md, then wire a "
                    "ccxt.coinbaseinternational client + set enabled=True."
                )
                self._warned_disabled = True
            return None

        try:
            symbol = to_venue_symbol(asset, "coinbase", market)
        except KeyError as e:
            logger.warning(f"[COINBASE_FUNDING] symbol map miss for {asset}: {e}")
            return None

        try:
            # NOTE: ccxt.coinbaseinternational reports fetchFundingRate=False,
            # so a real implementation calls the venue's raw/implicit funding
            # endpoint here (e.g. instruments/{symbol}/funding) and parses the
            # period + rate. Left unimplemented on purpose until the product is
            # confirmed; raising here would break fail-closed, so we return None.
            raw = await self._raw_funding(symbol)
            if not raw:
                return None
            period = float(raw.get("period_hours", _DEFAULT_FUNDING_PERIOD_HOURS))
            rate = float(raw["funding_rate"])
            return FundingRateData(
                symbol=symbol,
                funding_rate=rate,
                funding_period_hours=period,
                funding_rate_8h=self._normalize_to_8h(rate, period),
                next_funding_ts=raw.get("next_funding_ts"),
                venue="coinbase",
            )
        except Exception as e:  # fail-closed
            logger.warning(
                f"[COINBASE_FUNDING] fetch failed for {asset}/{symbol}: "
                f"{type(e).__name__}: {e}; falling back to Kraken funding"
            )
            return None

    async def _raw_funding(self, symbol: str) -> Optional[dict]:
        """Raw funding endpoint call. Unimplemented scaffold — returns None so
        the feed stays fail-closed. Implement against the confirmed product's
        API (Phase 2 step 2.3) once access is verified."""
        return None

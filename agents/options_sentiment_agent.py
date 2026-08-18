"""
[OPTIONS-AGENT] Options Sentiment Agent - CoinGlass Options Data
================================================================
Authority Level: ADVISE (no veto, no trigger)

Signals:
  1. put_call_ratio (OI-based): >1.3 = institutional hedging = short confirmation
                                <0.6 = greed = contrarian short
  2. max_pain_distance: price above max_pain = gravity pull down = short tailwind
  3. options_futures_ratio: rising = smart money share increasing
  4. volume_put_call: cross-validates OI ratio

Data Source: CoinGlass API (open-api-v3.coinglass.com)
Frequency: Every 4H tick
Fail-safe: All errors -> neutral output (short_confirmation=0, confidence=0)
"""

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Safe import of shared HTTP helper
try:
    from data_mgmt.feeds._http import create_session
except ImportError:
    import aiohttp
    def create_session(**kwargs):
        return aiohttp.ClientSession(**kwargs)


class OptionsSentimentAgent:
    """
    [OPTIONS-AGENT] CoinGlass-based options sentiment for short-biased system.

    Authority: ADVISE only - feeds into Authority Fusion, never vetoes.
    """

    BASE_URL = "https://open-api-v3.coinglass.com/api"

    # [P218] Endpoints reported dead, warned once each. A 404 on a HARDCODED
    # url is not a market condition — it is a broken integration — but the
    # handler below returned None and the caller then used its neutral default
    # (put_call_ratio = 1.0). "The API path no longer exists" and "options
    # sentiment is perfectly balanced" produced byte-identical output, which is
    # why this sat unnoticed. Probed live 2026-08-07: /option/info/max-pain,
    # /option/info/oi and /option/info/volume ALL return HTTP 404, while the
    # parent /option/info returns real data — so this is a v3 API path change,
    # NOT a subscription tier limit (the same key serves funding/OI/liquidations
    # fine, and I had guessed "tier limit" before checking).
    _DEAD_ENDPOINTS_WARNED: set = set()

    # [P308] Paths that answered 404 THIS PROCESS. The URL is a hardcoded
    # constant, so a 404 is a permanent property of the integration, not a
    # market condition — re-requesting it every tick can never succeed. P218
    # latched the WARNING and left the CALL in place, so this has been
    # spending paid CoinGlass quota on three dead endpoints x BTC/ETH on
    # every tick since. Process-scoped deliberately: a redeploy re-probes, so
    # if CoinGlass ever restores the path we find out on the next restart
    # rather than never.
    #
    # ONLY 404. A 5xx or a 429 is transient and must keep retrying — skipping
    # those would turn a venue blip into a permanently dark feed, which is the
    # opposite failure and a worse one.
    _DEAD_ENDPOINTS_404: set = set()

    @classmethod
    def _is_known_dead(cls, path: str) -> bool:
        return path in cls._DEAD_ENDPOINTS_404

    def _report_http(self, path: str, status: int) -> None:
        """Warn once per (path, status) when an endpoint answers non-200."""
        key = f"{path}:{status}"
        if status == 404:
            # [P308] Registered whether or not we have already warned: the
            # warn-latch fires once, the skip must apply on every later tick.
            self._DEAD_ENDPOINTS_404.add(path)
        if key in self._DEAD_ENDPOINTS_WARNED:
            return
        self._DEAD_ENDPOINTS_WARNED.add(key)
        if status == 404:
            logger.warning(
                f"[OPTIONS-AGENT] {path} -> HTTP 404: this endpoint does not "
                f"exist. The agent will fall back to put_call_ratio=1.0 "
                f"(NEUTRAL) forever, which is indistinguishable from a balanced "
                f"options market. NOT a tier limit — the same CoinGlass key "
                f"serves funding/OI fine and /option/info returns data."
            )
        else:
            logger.warning(f"[OPTIONS-AGENT] {path} -> HTTP {status}")
    SUPPORTED_ASSETS = ("BTC", "ETH")  # Options data mainly BTC/ETH

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        self._mock_mode = not bool(self._api_key)

        # History for z-score computation (42 bars = 7 days at 4H)
        self._pcr_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=42))
        self._max_pain_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=42))

        # Cache
        self._last_result: Dict[str, Dict] = {}
        self._last_fetch_time: float = 0
        self._fetch_errors: int = 0

        if self._mock_mode:
            logger.info("[OPTIONS-AGENT] No API key -> mock mode")
        else:
            logger.info("[OPTIONS-AGENT] Initialized with CoinGlass API")

    def _headers(self) -> Dict[str, str]:
        return {
            "CG-API-KEY": self._api_key,
            "accept": "application/json",
        }

    async def generate_signal(self, asset: str, current_price: float = 0.0) -> Dict[str, Any]:
        """
        [OPTIONS-AGENT] Generate options sentiment signal for asset.

        Returns neutral values on any error - never blocks tick loop.
        """
        neutral = {
            "put_call_ratio": 1.0,
            "put_call_zscore": 0.0,
            "max_pain_price": 0.0,
            "max_pain_distance_pct": 0.0,
            "options_futures_oi_ratio": 0.0,
            "short_confirmation": 0.0,
            "confidence": 0.0,
            "source": "options_sentiment",
        }

        if asset not in self.SUPPORTED_ASSETS:
            # SOL has minimal options data
            return neutral

        try:
            if self._mock_mode:
                return self._mock_signal(asset)

            return await self._fetch_and_compute(asset, current_price)

        except Exception as e:
            self._fetch_errors += 1
            logger.debug(f"[OPTIONS-AGENT] {asset} error: {e}")
            cached = self._last_result.get(asset)
            return cached if cached else neutral

    async def _fetch_and_compute(self, asset: str, current_price: float) -> Dict[str, Any]:
        """Fetch all endpoints and compute composite signal."""
        async with create_session() as session:
            # Parallel fetch all endpoints
            max_pain_task = self._fetch_max_pain(session, asset)
            oi_task = self._fetch_option_oi(session, asset)
            volume_task = self._fetch_option_volume(session, asset)

            max_pain_data, oi_data, volume_data = await asyncio.gather(
                max_pain_task, oi_task, volume_task,
                return_exceptions=True,
            )

        # Process results (handle exceptions from gather)
        max_pain_price = 0.0
        if not isinstance(max_pain_data, Exception) and max_pain_data:
            max_pain_price = max_pain_data.get("max_pain", 0.0)

        pcr_oi = 1.0
        if not isinstance(oi_data, Exception) and oi_data:
            pcr_oi = oi_data.get("put_call_ratio", 1.0)

        pcr_vol = 1.0
        if not isinstance(volume_data, Exception) and volume_data:
            pcr_vol = volume_data.get("put_call_ratio", 1.0)

        # Update history for z-score
        if pcr_oi != 1.0:
            self._pcr_history[asset].append(pcr_oi)
        if max_pain_price > 0:
            self._max_pain_history[asset].append(max_pain_price)

        # Compute z-score
        pcr_zscore = self._compute_zscore(pcr_oi, self._pcr_history[asset])

        # Max pain distance
        price = current_price if current_price > 0 else max_pain_price
        mp_distance = 0.0
        if price > 0 and max_pain_price > 0:
            mp_distance = (price - max_pain_price) / max_pain_price  # [FIX-AG14] Was /price — distorts signal across different price ranges. /max_pain gives consistent % deviation.

        # Composite short confirmation
        short_conf = self._compute_short_confirmation(pcr_oi, pcr_zscore, mp_distance, pcr_vol)

        # Confidence based on data availability
        data_points = sum([
            pcr_oi != 1.0,
            max_pain_price > 0,
            pcr_vol != 1.0,
        ])
        confidence = min(0.8, data_points * 0.25)

        result = {
            "put_call_ratio": round(pcr_oi, 3),
            "put_call_zscore": round(pcr_zscore, 3),
            "max_pain_price": round(max_pain_price, 2),
            "max_pain_distance_pct": round(mp_distance, 4),
            "options_futures_oi_ratio": 0.0,
            "volume_put_call_ratio": round(pcr_vol, 3),
            "short_confirmation": round(short_conf, 3),
            "confidence": round(confidence, 3),
            "source": "options_sentiment",
        }

        self._last_result[asset] = result
        self._last_fetch_time = time.time()
        return result

    def apply_external_pcr(
        self,
        asset: str,
        pcr_oi: float,
        pcr_vol: Optional[float] = None,
        current_price: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """[P293] Build a signal from an externally-sourced put/call ratio.

        The CoinGlass v3 option paths this agent was written against all
        return 404 (P218), so its own fetch can only ever produce the
        neutral `put_call_ratio=1.0`. Deribit — where the crypto options
        open interest actually sits — publishes the book for free, so the
        ratio is available; only the SOURCE is different.

        The conversion (z-score, thresholds, composite) deliberately lives
        HERE and is reused verbatim rather than reimplemented at the call
        site, so an external PCR and a native one can never be scored by
        two different rules (P172 single-source).

        Returns None — never a neutral dict — when the input is unusable,
        so the caller can leave the previous reading alone instead of
        overwriting it with a fabricated balance.

        NOTE max_pain is genuinely unavailable (no working endpoint), so
        mp_distance stays 0.0 and contributes nothing. That is an honest
        absence, and `confidence` reflects it by counting fewer inputs.
        """
        asset = (asset or "").upper()
        if asset not in self.SUPPORTED_ASSETS:
            return None
        try:
            pcr = float(pcr_oi)
        except (TypeError, ValueError):  # noqa: silent-swallow
            # [P293] A non-numeric PCR is not a measurement; None lets the
            # caller keep the previous reading instead of overwriting it.
            return None
        if not (pcr > 0.0) or pcr != pcr:
            return None

        try:
            pvol = float(pcr_vol) if pcr_vol is not None else 1.0
            if not (pvol > 0.0) or pvol != pvol:
                pvol = 1.0
        except (TypeError, ValueError):
            pvol = 1.0

        self._pcr_history[asset].append(pcr)
        pcr_z = self._compute_zscore(pcr, self._pcr_history[asset])

        # No max-pain source exists; 0.0 means "this leg contributed nothing".
        mp_distance = 0.0
        short_conf = self._compute_short_confirmation(pcr, pcr_z, mp_distance, pvol)

        data_points = sum([pcr != 1.0, pvol != 1.0])
        confidence = min(0.8, data_points * 0.25)

        result = {
            "put_call_ratio": round(pcr, 3),
            "put_call_zscore": round(pcr_z, 3),
            "max_pain_price": 0.0,
            "max_pain_distance_pct": 0.0,
            "options_futures_oi_ratio": 0.0,
            "volume_put_call_ratio": round(pvol, 3),
            "short_confirmation": round(short_conf, 3),
            "confidence": round(confidence, 3),
            "source": "options_sentiment_deribit",
        }
        self._last_result[asset] = result
        return result

    def _compute_short_confirmation(
        self, pcr_oi: float, pcr_zscore: float, mp_distance: float, pcr_vol: float,
    ) -> float:
        """
        Composite short confirmation score (-1.0 to +1.0).
        Positive = short tailwind, Negative = short headwind.
        """
        scores = []

        # 1. Put/Call ratio (OI): >1.3 = institutional buying puts = short friendly
        if pcr_oi > 1.3:
            scores.append(min(1.0, (pcr_oi - 1.0) / 0.5))  # 1.0->0, 1.5->1.0
        elif pcr_oi < 0.6:
            scores.append(min(0.5, (1.0 - pcr_oi) / 0.8))  # Contrarian: low PCR = complacency
        else:
            scores.append(0.0)

        # 2. PCR z-score: elevated = unusual put buying
        if pcr_zscore > 1.0:
            scores.append(min(1.0, pcr_zscore / 3.0))
        elif pcr_zscore < -1.0:
            scores.append(max(-0.5, pcr_zscore / 4.0))  # Low PCR = mild contrarian
        else:
            scores.append(0.0)

        # 3. Max pain distance: price above max_pain = gravity pull = short tailwind
        if mp_distance > 0.03:
            scores.append(min(1.0, mp_distance / 0.10))  # 3%->0.3, 10%->1.0
        elif mp_distance < -0.03:
            scores.append(max(-0.5, mp_distance / 0.10))  # Below max pain = support
        else:
            scores.append(0.0)

        # 4. Volume PCR cross-validation
        if pcr_vol > 1.3 and pcr_oi > 1.0:
            scores.append(0.3)  # Both elevated = strong signal
        elif pcr_vol < 0.5:
            scores.append(-0.2)  # Low volume PCR = call dominance
        else:
            scores.append(0.0)

        return np.clip(np.mean(scores), -1.0, 1.0) if scores else 0.0

    @staticmethod
    def _compute_zscore(value: float, history: deque) -> float:
        if len(history) < 5:
            return 0.0
        arr = np.array(list(history))
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-8:
            return 0.0
        return float((value - mean) / std)

    async def _fetch_max_pain(self, session, asset: str) -> Optional[Dict]:
        """GET /api/option/info/max-pain?symbol={asset}"""
        try:
            url = f"{self.BASE_URL}/option/info/max-pain"
            if self._is_known_dead(url.rsplit("/api", 1)[-1]):
                return None            # [P308] 404 already proven this process
            async with session.get(
                url, headers=self._headers(),
                params={"symbol": asset},
                timeout=__import__("aiohttp").ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    self._report_http(url.rsplit("/api", 1)[-1], resp.status)
                    return None
                data = await resp.json()
                if str(data.get("code")) != "0" or not data.get("data"):
                    return None
                # Extract nearest expiry max pain
                entries = data["data"]
                if isinstance(entries, list) and entries:
                    return {"max_pain": float(entries[0].get("maxPain", 0))}
                elif isinstance(entries, dict):
                    return {"max_pain": float(entries.get("maxPain", 0))}
                return None
        except Exception as e:
            logger.debug(f"[OPTIONS-AGENT] max_pain fetch error: {e}")
            return None

    async def _fetch_option_oi(self, session, asset: str) -> Optional[Dict]:
        """GET /api/option/info/oi -> aggregate put vs call OI."""
        try:
            url = f"{self.BASE_URL}/option/info/oi"
            if self._is_known_dead(url.rsplit("/api", 1)[-1]):
                return None            # [P308] 404 already proven this process
            async with session.get(
                url, headers=self._headers(),
                params={"symbol": asset},
                timeout=__import__("aiohttp").ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    self._report_http(url.rsplit("/api", 1)[-1], resp.status)
                    return None
                data = await resp.json()
                if str(data.get("code")) != "0" or not data.get("data"):
                    return None
                d = data["data"]
                total_put = float(d.get("putOI", d.get("putOpenInterest", 0)))
                total_call = float(d.get("callOI", d.get("callOpenInterest", 0)))
                if total_call > 0:
                    return {"put_call_ratio": total_put / total_call}
                return None
        except Exception as e:
            logger.debug(f"[OPTIONS-AGENT] option OI fetch error: {e}")
            return None

    async def _fetch_option_volume(self, session, asset: str) -> Optional[Dict]:
        """GET /api/option/info/volume -> put vs call volume."""
        try:
            url = f"{self.BASE_URL}/option/info/volume"
            if self._is_known_dead(url.rsplit("/api", 1)[-1]):
                return None            # [P308] 404 already proven this process
            async with session.get(
                url, headers=self._headers(),
                params={"symbol": asset},
                timeout=__import__("aiohttp").ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    self._report_http(url.rsplit("/api", 1)[-1], resp.status)
                    return None
                data = await resp.json()
                if str(data.get("code")) != "0" or not data.get("data"):
                    return None
                d = data["data"]
                put_vol = float(d.get("putVolume", 0))
                call_vol = float(d.get("callVolume", 0))
                if call_vol > 0:
                    return {"put_call_ratio": put_vol / call_vol}
                return None
        except Exception as e:
            logger.debug(f"[OPTIONS-AGENT] option volume fetch error: {e}")
            return None

    def _mock_signal(self, asset: str) -> Dict[str, Any]:
        """[OPTIONS-AGENT] Mock = neutral, no signal."""
        return {
            "put_call_ratio": 1.0,
            "put_call_zscore": 0.0,
            "max_pain_price": 0.0,
            "max_pain_distance_pct": 0.0,
            "options_futures_oi_ratio": 0.0,
            "volume_put_call_ratio": 1.0,
            "short_confirmation": 0.0,
            "confidence": 0.0,
            "source": "options_sentiment_mock",
        }


# =============================================================================
# SINGLETON
# =============================================================================

_instance: Optional[OptionsSentimentAgent] = None


def get_options_sentiment_agent(api_key: Optional[str] = None) -> OptionsSentimentAgent:
    """Get or create singleton."""
    global _instance
    if _instance is None:
        _instance = OptionsSentimentAgent(api_key=api_key)
    else:
        resolved_api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        if getattr(_instance, "_api_key", "") != resolved_api_key:
            logger.info("[OPTIONS-AGENT] API key changed; refreshing singleton instance")
            _instance = OptionsSentimentAgent(api_key=api_key)
    return _instance


def reset_options_sentiment_agent():
    """Reset singleton."""
    global _instance
    _instance = None

"""
MarketDataPipeline - fetches live data + computes TA indicators + GMM regime.

Extracted from main.py (Phase 4C) to reduce God Object.
Owns: _ta_cache, _wavelet_buffers, _depth_history, asset_returns_cache,
      _regime_smoother_state.
"""

import asyncio
import importlib.util
import logging
import math
import time as _time
from collections import deque
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RUNTIME_FEATURE_ENGINEER = None
_RUNTIME_FEATURE_ENGINEER_FAILED = False


def _get_runtime_feature_engineer():
    """Load the training FeatureEngineer once so runtime matches training units."""
    global _RUNTIME_FEATURE_ENGINEER, _RUNTIME_FEATURE_ENGINEER_FAILED

    if _RUNTIME_FEATURE_ENGINEER is not None:
        return _RUNTIME_FEATURE_ENGINEER
    if _RUNTIME_FEATURE_ENGINEER_FAILED:
        return None

    try:
        features_path = Path(__file__).resolve().parent.parent / "training" / "drl" / "features.py"
        if not features_path.exists():
            raise FileNotFoundError(features_path)
        spec = importlib.util.spec_from_file_location("_training_drl_features_runtime", str(features_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for {features_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RUNTIME_FEATURE_ENGINEER = module.FeatureEngineer(state_dim=128, cross_asset_dim=16)
        logger.info("[MarketDataPipeline] Runtime FeatureEngineer loaded")
    except Exception as exc:
        _RUNTIME_FEATURE_ENGINEER_FAILED = True
        logger.warning(f"[MarketDataPipeline] FeatureEngineer unavailable: {exc}")

    return _RUNTIME_FEATURE_ENGINEER

# VPIN calculator (batch from Kraken REST trades)
_VPIN_AVAILABLE = False
try:
    from market.vpin_calculator import BatchVPINCalculator
    _VPIN_AVAILABLE = True
except ImportError:
    pass

# WhaleDetector (3-dim exchange-level large order detection)
_WHALE_DETECTOR_AVAILABLE = False
try:
    from agents.whale_detector import get_whale_detector
    _WHALE_DETECTOR_AVAILABLE = True
except ImportError:
    pass

# TA indicators (optional import, same as main.py)
try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("[MarketDataPipeline] ta library unavailable")


# ─── Regime fitness constants (moved from main.py class body) ──────────────
REGIME_SIGNAL_WEIGHTS = {
    "MOMENTUM_RALLY":      {"ema": 0.35, "macd": 0.30, "rsi": 0.15, "bb": 0.20},
    "PANIC_SELLOFF":       {"ema": 0.30, "macd": 0.30, "rsi": 0.20, "bb": 0.20},
    "QUIET_ACCUMULATION":  {"ema": 0.15, "macd": 0.20, "rsi": 0.30, "bb": 0.35},
    "MEAN_REVERT":         {"ema": 0.10, "macd": 0.15, "rsi": 0.40, "bb": 0.35},
    "VOLATILE_CHOP":       {"ema": 0.15, "macd": 0.20, "rsi": 0.35, "bb": 0.30},
    "EXTREME_VOLATILITY":  {"ema": 0.15, "macd": 0.20, "rsi": 0.35, "bb": 0.30},
    "WEAK_CONSOLIDATION":  {"ema": 0.20, "macd": 0.20, "rsi": 0.30, "bb": 0.30},
}

REGIME_STRATEGY_FIT = {
    "mean_revert": {
        "MEAN_REVERT": 1.3, "QUIET_ACCUMULATION": 1.3,
        "WEAK_CONSOLIDATION": 0.7,  # [FIX-MR-RSI-CONFIRM] was 1.1. WEAK_CONSOLIDATION
        # is ambiguous — price can trend, not just revert. 0.7 reduces mean_revert
        # dominance, allowing momentum/other strategies to compete when signals align.
        "NEUTRAL_DRIFT": 0.9, "STEADY_UPTREND": 0.4,
        "_default": 0.5,
    },
    "momentum": {
        "MOMENTUM_RALLY": 1.3, "PANIC_SELLOFF": 1.3, "STEADY_UPTREND": 1.1,
        "NEUTRAL_DRIFT": 0.5, "QUIET_ACCUMULATION": 0.3, "WEAK_CONSOLIDATION": 0.3,
        "_default": 0.35,
    },
    "volume_breakout": {
        "MOMENTUM_RALLY": 1.2, "PANIC_SELLOFF": 1.2, "VOLATILE_CHOP": 1.1,
        "EXTREME_VOLATILITY": 0.9,
        "_default": 0.45,
    },
    "vrp": {
        "EXTREME_VOLATILITY": 1.3, "VOLATILE_CHOP": 1.2,
        "PANIC_SELLOFF": 0.9,
        "_default": 0.4,
    },
    # [2026-04-08] HOLD: no-trade strategy for choppy/directionless markets.
    # Prevents forced entries when no strategy has edge. Wins Best-of-N when
    # ADX<15, RSI neutral, volume thin — exactly the conditions that produced
    # 14.3% SOL win rate and 12.5% SHORT win rate in paper run.
    "hold": {
        "QUIET_ACCUMULATION": 0.8, "WEAK_CONSOLIDATION": 0.8,
        "NEUTRAL_DRIFT": 0.7, "VOLATILE_CHOP": 0.6,
        "STEADY_UPTREND": 0.15, "MOMENTUM_RALLY": 0.10,
        "PANIC_SELLOFF": 0.10, "EXTREME_VOLATILITY": 0.3,
        "_default": 0.3,
    },
}

# L4-13: Default strength for unknown regime labels (conservative)
_UNKNOWN_REGIME_FIT = 0.5

# All known regime labels across strategy fit tables
_KNOWN_REGIMES = set()
for _fits in REGIME_STRATEGY_FIT.values():
    _KNOWN_REGIMES.update(_fits.keys())
# Also include regimes from other config tables
_KNOWN_REGIMES.update({
    "STEADY_UPTREND", "NEUTRAL_DRIFT",  # Named in GMM config
})
_LOGGED_UNKNOWN_REGIMES: set = set()


class MarketDataPipeline:
    """Fetches live Kraken data, computes TA indicators, GMM regime, quant signals."""

    def __init__(
        self,
        *,
        # Threading
        sync_executor,
        assets: List[str],
        # Exchange
        kraken_rest=None,
        ccxt_exchange=None,
        # Services (all optional)
        orderbook_analyzer=None,
        weekend_manager=None,
        trade_gate=None,
        p0_integrator=None,
        correlation_monitor=None,
        adaptive_stop=None,
        sq_tracker=None,
        # GMM models
        gmm_models: Dict = None,
        gmm_model=None,
        gmm_configs: Dict = None,
        gmm_scaler_mean=None,
        gmm_scaler_scale=None,
        gmm_regime_names: list = None,
        gmm_feature_cols: list = None,
        # Config
        regime_smoother_persistence: int = 2,
        # Shared mutable state (dict references - mutations visible to runner)
        paper_positions: Dict = None,
    ):
        # Exchange
        self._sync_executor = sync_executor
        self._kraken_rest = kraken_rest
        self._ccxt_exchange = ccxt_exchange

        # Services
        self._orderbook_analyzer = orderbook_analyzer
        self._weekend_manager = weekend_manager
        self._trade_gate = trade_gate
        self._p0_integrator = p0_integrator
        self._correlation_monitor = correlation_monitor
        self._adaptive_stop = adaptive_stop
        self._sq_tracker = sq_tracker

        # GMM
        self._gmm_models = gmm_models or {}
        self._gmm_model = gmm_model
        self._gmm_configs = gmm_configs or {}
        self._gmm_scaler_mean = gmm_scaler_mean
        self._gmm_scaler_scale = gmm_scaler_scale
        self._gmm_regime_names = gmm_regime_names or []
        self._gmm_feature_cols = gmm_feature_cols or []

        # Config
        self._regime_smoother_persistence = regime_smoother_persistence

        # Shared state (dict reference - same object as runner's _paper_positions)
        self._paper_positions = paper_positions if paper_positions is not None else {}

        # Dynamic state updated by runner each round
        self.current_drawdown_pct: float = 0.0

        # ── Owned mutable state ──
        self._ta_cache: Dict[str, Dict] = {}

        # OFI z-score: rolling history of order_book_imbalance per asset (42 bars = 7 days at 4H)
        self._ofi_history: Dict[str, deque] = {a: deque(maxlen=42) for a in assets}

        _wavelet_targets = [
            "rsi_14",
            "macd_12_26",
            "bb_width_20",
            "atr_14",
            "vol_ratio_s",
        ]
        self._wavelet_buffers: Dict[str, Dict[str, deque]] = {
            a: {feat: deque(maxlen=256) for feat in _wavelet_targets}
            for a in assets
        }

        self._depth_history: Dict[str, deque] = {
            a: deque(maxlen=120) for a in assets
        }
        self._bootstrap_ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._binance_vision_recent_cache: Dict[str, Dict[str, Any]] = {}
        self._requests_session = None
        # [2026-04-10] CCXT session auto-reset: after N consecutive fetch failures
        # across ALL assets, recreate the CCXT exchange object to clear stale
        # SSL sessions and connection pools. Without this, a stuck TCP connection
        # causes 21h+ data blackout (process alive but zero data).
        self._global_fetch_failure_count: int = 0
        self._global_fetch_success_count: int = 0
        self._ccxt_reset_count: int = 0
        self._CCXT_RESET_THRESHOLD: int = 10  # reset after 10 consecutive failures (~5 min)
        # Orderbook resiliency state (avoid noisy risk triggers on transient fetch failures)
        self._last_orderbook_depth_usd: Dict[str, float] = {}
        self._last_orderbook_imbalance: Dict[str, float] = {}
        self._last_orderbook_ts: Dict[str, float] = {}
        self._orderbook_failure_streak: Dict[str, int] = {a: 0 for a in assets}
        self._orderbook_stale_ttl_seconds: float = 180.0

        # Public so runner's lead-lag code can read it
        self.asset_returns_cache: Dict[str, Any] = {}

        self._regime_smoother_state: Dict[str, Dict] = {}

        # VPIN calculator (batch from Kraken REST trades)
        self._vpin_calc = None
        if _VPIN_AVAILABLE:
            try:
                self._vpin_calc = BatchVPINCalculator(
                    bucket_size_usd=10_000.0, n_buckets=30,
                )
            except Exception as e:
                logger.warning(f"BatchVPINCalculator init failed: {e}")

    # ─── Static helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_weekend_now() -> bool:
        now = datetime.now(timezone.utc)
        wd = now.weekday()
        h = now.hour
        if wd == 4 and h >= 21:
            return True
        if wd == 5:
            return True
        if wd == 6 and h < 21:
            return True
        return False

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(result):
            return float(default)
        return result

    def _compute_runtime_feature_snapshot(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build the normalized feature subset used by training-time denoising."""
        engineer = _get_runtime_feature_engineer()
        if engineer is None or df is None or len(df) == 0:
            return {}

        try:
            featured = engineer.compute_features(df)
            if featured is None or len(featured) == 0:
                return {}
            last_row = featured.iloc[-1]
            return {
                "rsi_14": self._safe_float(last_row.get("rsi_14", 0.0)),
                "macd_12_26": self._safe_float(last_row.get("macd_12_26", 0.0)),
                "bb_width_20": self._safe_float(last_row.get("bb_width_20", 0.0)),
                "atr_14": self._safe_float(last_row.get("atr_14", 0.0)),
                "vol_ratio_s": self._safe_float(last_row.get("vol_ratio_s", 0.0)),
            }
        except Exception as exc:
            logger.debug(f"[PREPARE_DATA] runtime feature snapshot failed: {exc}")
            return {}

    def _reset_ccxt_session(self):
        """Recreate CCXT exchange objects to clear stale SSL/connection pool.
        Called after _CCXT_RESET_THRESHOLD consecutive fetch failures.
        Resets both the primary (kraken_rest._exchange) and fallback (_ccxt_exchange)."""
        self._ccxt_reset_count += 1
        logger.warning(
            f"[CCXT_RESET] Resetting CCXT session after {self._global_fetch_failure_count} "
            f"consecutive failures (reset #{self._ccxt_reset_count})"
        )
        self._global_fetch_failure_count = 0
        try:
            import ccxt
            # Reset primary: kraken_rest._exchange (authenticated, used for data+trading)
            if self._kraken_rest and hasattr(self._kraken_rest, '_exchange'):
                old_ex = self._kraken_rest._exchange
                if old_ex:
                    try:
                        if hasattr(old_ex, 'session') and old_ex.session:
                            old_ex.session.close()
                    except Exception:
                        pass
                    # Recreate with same config (preserves API keys)
                    new_cfg = {
                        'timeout': getattr(old_ex, 'timeout', 10000),
                        'enableRateLimit': getattr(old_ex, 'enableRateLimit', True),
                    }
                    if getattr(old_ex, 'apiKey', None):
                        new_cfg['apiKey'] = old_ex.apiKey
                        new_cfg['secret'] = old_ex.secret
                    self._kraken_rest._exchange = ccxt.kraken(new_cfg)
                    logger.info("[CCXT_RESET] Primary kraken_rest._exchange recreated")

            # Reset fallback: _ccxt_exchange (unauthenticated)
            if self._ccxt_exchange:
                try:
                    if hasattr(self._ccxt_exchange, 'session') and self._ccxt_exchange.session:
                        self._ccxt_exchange.session.close()
                except Exception:
                    pass
            self._ccxt_exchange = ccxt.kraken({
                'timeout': 10000,
                'enableRateLimit': True,
            })
            logger.info("[CCXT_RESET] Fallback _ccxt_exchange recreated")
        except Exception as e:
            logger.error(f"[CCXT_RESET] Failed to recreate CCXT session: {e}")

        # Also reset requests session (used by binance vision etc)
        if self._requests_session:
            try:
                self._requests_session.close()
            except Exception:
                pass
            self._requests_session = None

    def _get_requests_session(self):
        if self._requests_session is not None:
            return self._requests_session
        try:
            import requests
        except ImportError:
            return None
        self._requests_session = requests.Session()
        self._requests_session.headers.update({"User-Agent": "HMATS/MarketDataPipeline"})
        return self._requests_session

    @staticmethod
    def _ohlcv_bars_to_frame(ohlcv_bars: List[List[float]]) -> pd.DataFrame:
        if not ohlcv_bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"])

        df = pd.DataFrame(
            ohlcv_bars,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce").astype("Int64")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp_ms", "open", "high", "low", "close", "volume"]).copy()
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"])
        df["timestamp_ms"] = df["timestamp_ms"].astype("int64")
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df = df.sort_values("timestamp_ms").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"]]

    def _get_bootstrap_ohlcv_frame(self, asset: str) -> pd.DataFrame:
        cache_df = self._bootstrap_ohlcv_cache.get(asset)
        if cache_df is not None:
            return cache_df

        try:
            raw_path = Path(__file__).resolve().parent.parent / "training" / "training_data" / "raw" / f"{asset}_60m.parquet"
            if not raw_path.exists():
                self._bootstrap_ohlcv_cache[asset] = pd.DataFrame()
                return self._bootstrap_ohlcv_cache[asset]

            df = pd.read_parquet(raw_path)
            df.columns = df.columns.str.lower()
            if "volume_usdt" in df.columns:
                df = df.rename(columns={"volume_usdt": "volume"})
            elif "volume btc" in df.columns and "volume" not in df.columns:
                df = df.rename(columns={"volume btc": "volume"})
            elif f"volume {asset.lower()}" in df.columns and "volume" not in df.columns:
                df = df.rename(columns={f"volume {asset.lower()}": "volume"})

            if "unix_timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["unix_timestamp"], unit="ms", errors="coerce", utc=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            else:
                self._bootstrap_ohlcv_cache[asset] = pd.DataFrame()
                return self._bootstrap_ohlcv_cache[asset]

            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.set_index("timestamp")
            df_4h = df.resample("4h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna().reset_index()
            if not df_4h.empty:
                df_4h["timestamp_ms"] = (df_4h["timestamp"].astype("int64") // 1_000_000).astype("int64")
            self._bootstrap_ohlcv_cache[asset] = df_4h
            return df_4h
        except Exception as exc:
            logger.debug(f"[LIVE_DATA] {asset}: bootstrap OHLCV unavailable: {exc}")
            self._bootstrap_ohlcv_cache[asset] = pd.DataFrame()
            return self._bootstrap_ohlcv_cache[asset]

    @staticmethod
    def _normalize_binance_vision_ts(raw_ts: Any) -> Optional[int]:
        try:
            ts = int(raw_ts)
        except Exception:
            return None
        if ts > 10**15:
            ts //= 1000
        elif ts > 10**13:
            ts //= 1000
        return ts

    def _load_binance_vision_zip_frame(self, asset: str, url: str) -> pd.DataFrame:
        session = self._get_requests_session()
        if session is None:
            return pd.DataFrame()
        try:
            response = session.get(url, timeout=20)
            if response.status_code == 404:
                return pd.DataFrame()
            response.raise_for_status()
            with ZipFile(BytesIO(response.content)) as archive:
                members = archive.namelist()
                if not members:
                    return pd.DataFrame()
                with archive.open(members[0]) as handle:
                    raw_df = pd.read_csv(handle, header=None)
        except Exception as exc:
            logger.debug(f"[DRL_WARMUP] {asset}: fetch failed for {url}: {exc}")
            return pd.DataFrame()

        if raw_df.empty:
            return pd.DataFrame()

        raw_df = raw_df.iloc[:, :9].copy()
        raw_df.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume_base",
            "close_time",
            "volume_quote",
            "trade_count",
        ]
        raw_df["timestamp_ms"] = raw_df["open_time"].map(self._normalize_binance_vision_ts)
        raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp_ms"], unit="ms", utc=True, errors="coerce")
        for col in ["open", "high", "low", "close", "volume_quote"]:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
        raw_df = raw_df.dropna(subset=["timestamp_ms", "timestamp", "open", "high", "low", "close", "volume_quote"])
        if raw_df.empty:
            return pd.DataFrame()
        return raw_df.rename(columns={"volume_quote": "volume"})[
            ["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"]
        ].sort_values("timestamp_ms").reset_index(drop=True)

    def _get_binance_vision_recent_frame(self, asset: str) -> pd.DataFrame:
        today = datetime.now(timezone.utc).date()
        cached = self._binance_vision_recent_cache.get(asset)
        if cached and cached.get("loaded_for") == today:
            return cached.get("df", pd.DataFrame())

        symbol_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
        symbol = symbol_map.get(asset, f"{asset}USDT")
        base_df = self._get_bootstrap_ohlcv_frame(asset)
        if base_df is None or base_df.empty or "timestamp_ms" not in base_df.columns:
            self._binance_vision_recent_cache[asset] = {"loaded_for": today, "df": pd.DataFrame()}
            return self._binance_vision_recent_cache[asset]["df"]

        base_max_ts = int(base_df["timestamp_ms"].max())
        start_dt = pd.to_datetime(base_max_ts, unit="ms", utc=True) + timedelta(hours=4)
        frames: List[pd.DataFrame] = []

        month_cursor = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc).date()
        current_month = today.replace(day=1)
        while month_cursor < current_month:
            month_label = month_cursor.strftime("%Y-%m")
            url = (
                f"https://data.binance.vision/data/spot/monthly/klines/"
                f"{symbol}/4h/{symbol}-4h-{month_label}.zip"
            )
            frame = self._load_binance_vision_zip_frame(asset, url)
            if not frame.empty:
                frames.append(frame)
            month_cursor = (datetime(month_cursor.year + (month_cursor.month // 12), ((month_cursor.month % 12) + 1), 1, tzinfo=timezone.utc)).date()

        day_cursor = current_month
        while day_cursor <= today:
            day_label = day_cursor.strftime("%Y-%m-%d")
            url = (
                f"https://data.binance.vision/data/spot/daily/klines/"
                f"{symbol}/4h/{symbol}-4h-{day_label}.zip"
            )
            frame = self._load_binance_vision_zip_frame(asset, url)
            if not frame.empty:
                frames.append(frame)
            day_cursor += timedelta(days=1)

        if frames:
            recent_df = pd.concat(frames, ignore_index=True)
            recent_df = recent_df[recent_df["timestamp_ms"] > base_max_ts]
            recent_df = recent_df.drop_duplicates(subset=["timestamp_ms"], keep="last")
            recent_df = recent_df.sort_values("timestamp_ms").reset_index(drop=True)
        else:
            recent_df = pd.DataFrame()

        self._binance_vision_recent_cache[asset] = {"loaded_for": today, "df": recent_df}
        if not recent_df.empty:
            logger.info(
                f"[DRL_WARMUP] {asset}: Binance Vision tail loaded "
                f"({len(recent_df)} bars, {recent_df['timestamp'].iloc[0]} -> {recent_df['timestamp'].iloc[-1]})"
            )
        return recent_df

    def _build_drl_ohlcv(self, asset: str, kraken_ohlcv_raw: List[List[float]], target_bars: int) -> List[List[float]]:
        """Build DRL feature history using Binance-aligned data where possible."""
        kraken_df = self._ohlcv_bars_to_frame(kraken_ohlcv_raw)
        base_df = self._get_bootstrap_ohlcv_frame(asset)
        recent_df = self._get_binance_vision_recent_frame(asset)

        frames: List[pd.DataFrame] = []
        if base_df is not None and not base_df.empty:
            frames.append(base_df[["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"]].copy())
        if recent_df is not None and not recent_df.empty:
            frames.append(recent_df[["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"]].copy())

        if frames:
            drl_df = pd.concat(frames, ignore_index=True)
            drl_df = drl_df.drop_duplicates(subset=["timestamp_ms"], keep="last").sort_values("timestamp_ms").reset_index(drop=True)
            drl_last_ts = int(drl_df["timestamp_ms"].iloc[-1]) if not drl_df.empty else None
            if drl_last_ts is not None and not kraken_df.empty:
                kraken_tail = kraken_df[kraken_df["timestamp_ms"] > drl_last_ts]
                if not kraken_tail.empty:
                    drl_df = pd.concat([drl_df, kraken_tail], ignore_index=True)
                    drl_df = drl_df.drop_duplicates(subset=["timestamp_ms"], keep="last").sort_values("timestamp_ms").reset_index(drop=True)
        else:
            drl_df = kraken_df

        if drl_df is None or drl_df.empty:
            return kraken_ohlcv_raw

        if len(drl_df) < target_bars and not kraken_df.empty:
            combined = pd.concat([drl_df, kraken_df], ignore_index=True)
            drl_df = combined.drop_duplicates(subset=["timestamp_ms"], keep="last").sort_values("timestamp_ms").reset_index(drop=True)

        drl_df = drl_df.tail(target_bars).reset_index(drop=True)
        return [
            [
                int(row.timestamp_ms),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            ]
            for row in drl_df.itertuples(index=False)
        ]

    def _get_bootstrap_ohlcv(self, asset: str, *, before_ts_ms: Optional[int], needed_bars: int) -> List[List[float]]:
        """Backfill older 4H bars from local training parquet when exchange history is too short."""
        if needed_bars <= 0 or before_ts_ms is None:
            return []

        cache_df = self._get_bootstrap_ohlcv_frame(asset)
        if cache_df is None or cache_df.empty or "timestamp_ms" not in cache_df.columns:
            return []

        older = cache_df[cache_df["timestamp_ms"] < int(before_ts_ms)].tail(needed_bars)
        if older.empty:
            return []

        return [
            [
                int(row.timestamp_ms),
                float(row.open),
                float(row.high),
                float(row.low),
                float(row.close),
                float(row.volume),
            ]
            for row in older.itertuples(index=False)
        ]

    # ─── Public API ─────────────────────────────────────────────────────

    async def fetch_and_prepare(self, asset: str, tick_count: int = 0) -> Dict[str, Any]:
        """Fetch live data AND compute technical indicators.

        This is the combined replacement for main.py's _fetch_live_data +
        _prepare_market_data. Returns the enriched market_data dict.
        """
        raw = await self._fetch_live_data(asset)

        # Update TradeGate freshness timestamps
        if raw.get("data_valid"):
            _now = _time.time()
            if self._trade_gate:
                self._trade_gate.update_data_timestamp("price", _now)
                self._trade_gate.update_data_timestamp("orderbook", _now)
                self._trade_gate.update_data_timestamp("vpin", _now)
            _p0_tg = getattr(self._p0_integrator, 'trade_gate', None) if self._p0_integrator else None
            if _p0_tg:
                _p0_tg.update_data_timestamp("price", _now)
                _p0_tg.update_data_timestamp("orderbook", _now)
                _p0_tg.update_data_timestamp("vpin", _now)

        if not _TA_AVAILABLE:
            logger.warning("[PREPARE_DATA] ta library unavailable, returning raw data")
            return raw

        ohlcv_bars = raw.get("ohlcv_raw")
        drl_ohlcv_bars = raw.get("drl_ohlcv_raw") or ohlcv_bars
        if not drl_ohlcv_bars or len(drl_ohlcv_bars) < 30:
            drl_ohlcv_bars = ohlcv_bars

        if not ohlcv_bars or len(ohlcv_bars) < 30:
            logger.warning(
                f"[PREPARE_DATA] {asset}: insufficient OHLCV bars "
                f"({len(ohlcv_bars) if ohlcv_bars else 0}), skipping indicators"
            )
            return raw

        try:
            # [Phase 3] TA Cache - skip recomputation when OHLCV bars unchanged
            _last_bar_ts = ohlcv_bars[-1][0]
            _drl_last_bar_ts = drl_ohlcv_bars[-1][0] if drl_ohlcv_bars else _last_bar_ts
            _ta_cached = self._ta_cache.get(asset)
            _ta_hit = (_ta_cached is not None
                       and _ta_cached["last_ts"] == _last_bar_ts
                       and _ta_cached["n"] == len(ohlcv_bars)
                       and _ta_cached.get("drl_last_ts", _last_bar_ts) == _drl_last_bar_ts
                       and _ta_cached.get("drl_n", len(ohlcv_bars)) == len(drl_ohlcv_bars))

            if _ta_hit:
                df = _ta_cached["df"]
                drl_df = _ta_cached.get("drl_df", df)
                indicators = _ta_cached["indicators"]
                raw.update(_ta_cached["enrichment"])
                logger.debug(f"[PREPARE_DATA] {asset}: TA cache HIT (ts={_last_bar_ts})")

            if not _ta_hit:
                # Cache MISS - full DataFrame + TA + wavelet computation
                df = pd.DataFrame(
                    ohlcv_bars,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)

                if drl_ohlcv_bars == ohlcv_bars:
                    drl_df = df
                else:
                    drl_df = pd.DataFrame(
                        drl_ohlcv_bars,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    for col in ["open", "high", "low", "close", "volume"]:
                        drl_df[col] = drl_df[col].astype(float)

                close = df["close"]
                high = df["high"]
                low = df["low"]
                vol = df["volume"]

                # [WIRE-2] Feed latest OHLC bar to adaptive stop ATR calculator
                if self._adaptive_stop:
                    try:
                        self._adaptive_stop.atr_calculator.update(
                            asset, float(high.iloc[-1]), float(low.iloc[-1]),
                            float(close.iloc[-1])
                        )
                    except Exception as _as_err:
                        logger.debug(f"[WIRE-2] {asset}: adaptive stop ATR update skipped: {_as_err}")

                rsi_val = RSIIndicator(close, window=14).rsi().iloc[-1]
                macd_obj = MACD(close, window_slow=26, window_fast=12, window_sign=9)
                macd_val = macd_obj.macd().iloc[-1]
                macd_signal_val = macd_obj.macd_signal().iloc[-1]
                macd_hist_val = macd_obj.macd_diff().iloc[-1]

                bb = BollingerBands(close, window=20, window_dev=2)
                bb_upper = bb.bollinger_hband().iloc[-1]
                bb_middle = bb.bollinger_mavg().iloc[-1]
                bb_lower = bb.bollinger_lband().iloc[-1]
                bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0.0

                if len(close) >= 200:
                    ema_200 = EMAIndicator(close, window=200).ema_indicator().iloc[-1]
                else:
                    ema_200 = float("nan")
                    logger.debug(f"[SOTA L3] {asset}: only {len(close)} bars, EMA-200 unavailable")
                sma_50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]

                atr_series = AverageTrueRange(high, low, close, window=14).average_true_range()
                atr_val = atr_series.iloc[-1]
                atr_median_21 = atr_series.iloc[-21:].median() if len(atr_series) >= 21 else atr_series.median()
                atr_ratio = atr_val / atr_median_21 if atr_median_21 > 0 else 1.0

                adx_obj = ADXIndicator(high, low, close, window=14)
                adx_val = adx_obj.adx().iloc[-1]
                di_plus = adx_obj.adx_pos().iloc[-1]
                di_minus = adx_obj.adx_neg().iloc[-1]

                volume_sma = SMAIndicator(vol, window=20).sma_indicator().iloc[-1]

                indicators = {
                    "close": close.iloc[-1],
                    "ema_200": ema_200,
                    "bb_upper": bb_upper, "bb_middle": bb_middle,
                    "bb_lower": bb_lower, "bb_width": bb_width,
                    "rsi": rsi_val, "atr": atr_val,
                    "sma_50": sma_50, "volume_sma": volume_sma,
                    "macd": macd_val, "macd_signal": macd_signal_val,
                    "macd_hist": macd_hist_val,
                    "adx": adx_val, "di_plus": di_plus, "di_minus": di_minus,
                    "atr_ratio": atr_ratio,
                }

                for k, v in indicators.items():
                    if pd.isna(v):
                        indicators[k] = 0.0

                raw.update(indicators)
                raw.update(self._compute_runtime_feature_snapshot(drl_df))

                _cur_vol = float(vol.iloc[-1]) if len(vol) > 0 else 0
                _vol_sma = indicators.get("volume_sma", 0)
                _raw_vol_ratio = _cur_vol / _vol_sma if _vol_sma > 0 else 1.0
                raw["volume_ratio"] = _raw_vol_ratio

                _bar_progress_4h = 1.0
                _latest_bar_open_ts_ms = raw.get("latest_bar_open_ts_ms", 0)
                if _latest_bar_open_ts_ms:
                    try:
                        _bar_open_dt = datetime.fromtimestamp(
                            float(_latest_bar_open_ts_ms) / 1000.0, timezone.utc
                        )
                        _elapsed_s = max(
                            0.0,
                            (datetime.now(timezone.utc) - _bar_open_dt).total_seconds(),
                        )
                        _bar_progress_4h = min(1.0, _elapsed_s / (4.0 * 3600.0))
                    except Exception:
                        _bar_progress_4h = 1.0
                raw["bar_progress_4h"] = _bar_progress_4h
                if 0.0 < _bar_progress_4h < 1.0:
                    _progress_floor = max(_bar_progress_4h, 0.20)
                    _intrabar_pace = _raw_vol_ratio / _progress_floor
                    raw["volume_ratio_intrabar_pace"] = min(_intrabar_pace, 5.0)
                    raw["volume_ratio_effective"] = max(
                        _raw_vol_ratio, raw["volume_ratio_intrabar_pace"]
                    )
                else:
                    raw["volume_ratio_intrabar_pace"] = _raw_vol_ratio
                    raw["volume_ratio_effective"] = _raw_vol_ratio

                # [CRACK-1] Structure break: how far current price has broken prior 20-bar high/low
                _sb_lookback = min(20, len(high) - 1)  # exclude current bar
                if _sb_lookback >= 10:
                    _sb_high = float(high.iloc[-_sb_lookback - 1:-1].max())
                    _sb_low = float(low.iloc[-_sb_lookback - 1:-1].min())
                    _sb_cp = float(close.iloc[-1])
                    if _sb_cp > _sb_high and _sb_high > 0:
                        raw["structure_break_pct"] = (_sb_cp - _sb_high) / _sb_high
                    elif _sb_cp < _sb_low and _sb_low > 0:
                        raw["structure_break_pct"] = (_sb_low - _sb_cp) / _sb_low
                    else:
                        raw["structure_break_pct"] = 0.0
                    raw["structure_level"] = _sb_high if _sb_cp > _sb_high else _sb_low
                else:
                    raw["structure_break_pct"] = 0.0
                    raw["structure_level"] = 0.0

                _closes_20d = close.values[-120:] if len(close) >= 120 else close.values
                _c_min, _c_max = float(_closes_20d.min()), float(_closes_20d.max())
                raw["price_percentile_20d"] = (
                    (float(close.iloc[-1]) - _c_min) / (_c_max - _c_min + 1e-8)
                )

                logger.info(
                    f"[PREPARE_DATA] {asset}: RSI={indicators['rsi']:.1f} | "
                    f"MACD_hist={indicators['macd_hist']:.4f} | "
                    f"BB_width={indicators['bb_width']:.4f} | ADX={indicators['adx']:.1f}"
                )

                # ── Wavelet denoising ──
                _wv_map = {
                    "rsi_14": "rsi_14_denoised",
                    "macd_12_26": "macd_12_26_denoised",
                    "bb_width_20": "bb_width_20_denoised",
                    "atr_14": "atr_14_denoised",
                    "vol_ratio_s": "vol_ratio_s_denoised",
                }
                if asset in self._wavelet_buffers:
                    try:
                        from training.scripts.wavelet_denoise import wavelet_denoise as _wv_denoise
                        _wv_buf = self._wavelet_buffers[asset]
                        for src_name in _wv_map:
                            _wv_buf[src_name].append(float(raw.get(src_name, 0.0)))

                        for src_name, dst_name in _wv_map.items():
                            buf = _wv_buf[src_name]
                            if len(buf) >= 8:
                                denoised = _wv_denoise(np.array(buf))
                                raw[dst_name] = float(denoised[-1])
                            else:
                                raw[dst_name] = float(buf[-1]) if buf else 0.0
                    except Exception as _wv_err:
                        logger.warning(f"[WAVELET] {asset}: denoising FAILED ({_wv_err}) - using raw")
                        for _wv_src, _wv_dst in _wv_map.items():
                            raw[_wv_dst] = float(raw.get(_wv_src, 0.0))

                # Store TA cache
                _ta_enrich = {}
                _ta_enrich_keys = [
                    "close", "ema_200", "bb_upper", "bb_middle", "bb_lower",
                    "bb_width", "rsi", "atr", "sma_50", "volume_sma",
                    "macd", "macd_signal", "macd_hist", "adx", "di_plus",
                    "di_minus", "atr_ratio", "volume_ratio", "price_percentile_20d",
                    "rsi_14", "macd_12_26", "bb_width_20", "atr_14", "vol_ratio_s",
                    "rsi_14_denoised", "macd_12_26_denoised", "bb_width_20_denoised",
                    "atr_14_denoised", "vol_ratio_s_denoised",
                ]
                for _ek in _ta_enrich_keys:
                    if _ek in raw:
                        _ta_enrich[_ek] = raw[_ek]
                self._ta_cache[asset] = {
                    "last_ts": _last_bar_ts,
                    "n": len(ohlcv_bars),
                    "drl_last_ts": _drl_last_bar_ts,
                    "drl_n": len(drl_ohlcv_bars),
                    "df": df,
                    "drl_df": drl_df,
                    "indicators": dict(indicators),
                    "enrichment": _ta_enrich,
                }

            # ─── QUANT SIGNAL ───────────────────────────────────────────
            cur = indicators["close"]
            _rsi = indicators["rsi"]
            _macd_h = indicators["macd_hist"]
            _atr = indicators["atr"]
            _bb_u = indicators["bb_upper"]
            _bb_m = indicators["bb_middle"]
            _bb_l = indicators["bb_lower"]
            _ema = indicators["ema_200"]
            _adx = indicators["adx"]

            # [FIX-NAN-GUARD] TA libraries can return NaN on insufficient data
            # (RSI needs 14 bars, MACD 26, BB 20). A single NaN propagates through
            # signal chain → fusion → intent.direction = NaN → invalid orders.
            import math as _math
            _ta_vals = [_rsi, _macd_h, _atr, _bb_u, _bb_m, _bb_l, _ema, _adx]
            _ta_names = ["rsi", "macd_hist", "atr", "bb_upper", "bb_middle", "bb_lower", "ema_200", "adx"]
            _nan_count = 0
            for _tv, _tn in zip(_ta_vals, _ta_names):
                if _tv is None or (_math.isnan(_tv) if isinstance(_tv, float) else False) or (_math.isinf(_tv) if isinstance(_tv, float) else False):
                    _nan_count += 1
            if _nan_count > 0:
                # Replace NaN/None with safe defaults
                _rsi = float(_rsi) if (_rsi is not None and not _math.isnan(float(_rsi)) and not _math.isinf(float(_rsi))) else 50.0
                _macd_h = float(_macd_h) if (_macd_h is not None and not _math.isnan(float(_macd_h)) and not _math.isinf(float(_macd_h))) else 0.0
                _atr = float(_atr) if (_atr is not None and not _math.isnan(float(_atr)) and not _math.isinf(float(_atr))) else 0.01
                _bb_u = float(_bb_u) if (_bb_u is not None and not _math.isnan(float(_bb_u)) and not _math.isinf(float(_bb_u))) else cur * 1.02
                _bb_m = float(_bb_m) if (_bb_m is not None and not _math.isnan(float(_bb_m)) and not _math.isinf(float(_bb_m))) else cur
                _bb_l = float(_bb_l) if (_bb_l is not None and not _math.isnan(float(_bb_l)) and not _math.isinf(float(_bb_l))) else cur * 0.98
                _ema = float(_ema) if (_ema is not None and not _math.isnan(float(_ema)) and not _math.isinf(float(_ema))) else cur
                _adx = float(_adx) if (_adx is not None and not _math.isnan(float(_adx)) and not _math.isinf(float(_adx))) else 20.0
                logger.warning(f"[FIX-NAN-GUARD] {asset}: {_nan_count} NaN/None TA indicators replaced with defaults")

            rsi_sig = (50 - _rsi) / 50 if _rsi < 30 or _rsi > 70 else (_rsi - 50) / 100
            macd_sig = float(np.clip(_macd_h / (_atr * 2) if _atr > 0 else 0, -1, 1))

            bb_rng = _bb_u - _bb_l
            bb_pos = float(np.clip((cur - _bb_m) / (bb_rng / 2), -2.0, 2.0)) if bb_rng > 1e-8 else 0.0
            bb_sig = -bb_pos * 0.5 if abs(bb_pos) > 0.8 else bb_pos * 0.3

            if _ema > 0:
                ema_sig = float(np.clip((cur - _ema) / (_atr * 3 if _atr > 0 else cur * 0.03), -1, 1))
            else:
                ema_sig = 0.0

            adx_mult = 1.0 + max(0, _adx - 25) / 50 if _adx > 25 else (0.5 if _adx < 15 else 1.0)

            # ─── REGIME: GMM + smoother ─────────────────────────────────
            regime_conf = 0.40 + 0.45 * min(max(_adx - 20, 0) / 25, 1.0)
            gmm_regime_name = None
            _asset_upper = asset.upper() if isinstance(asset, str) else asset
            if _asset_upper in self._gmm_models or self._gmm_model is not None:
                try:
                    gmm_result = self._predict_gmm_regime(df, raw, np)
                    if gmm_result is not None:
                        regime_conf, gmm_regime_name = gmm_result
                except Exception as e:
                    logger.warning(f"[GMM] Prediction failed, using ADX proxy: {e}")
            raw["regime_confidence"] = regime_conf
            if gmm_regime_name:
                raw["regime_state"] = gmm_regime_name

            if not gmm_regime_name:
                if _adx > 30 and ema_sig > 0.3:
                    gmm_regime_name = "MOMENTUM_RALLY"
                elif _adx > 30 and ema_sig < -0.3:
                    gmm_regime_name = "PANIC_SELLOFF"
                elif _adx < 20 and abs(rsi_sig) < 0.2:
                    gmm_regime_name = "QUIET_ACCUMULATION"
                elif _rsi < 30 or _rsi > 70:
                    gmm_regime_name = "MEAN_REVERT"
                else:
                    gmm_regime_name = "WEAK_CONSOLIDATION"
                raw["regime_state"] = gmm_regime_name
                logger.info(f"[REGIME_FALLBACK] {asset}: ADX proxy -> {gmm_regime_name}")

            # RegimeSmoother
            _raw_regime = gmm_regime_name
            if gmm_regime_name and self._regime_smoother_persistence > 0:
                _state = self._regime_smoother_state.get(asset)
                if _state is None:
                    _state = {"current": gmm_regime_name, "pending": None, "count": 0}
                    self._regime_smoother_state[asset] = _state

                if gmm_regime_name == _state["current"]:
                    _state["pending"] = None
                    _state["count"] = 0
                elif gmm_regime_name == _state["pending"]:
                    _state["count"] += 1
                    if _state["count"] >= self._regime_smoother_persistence:
                        _state["current"] = gmm_regime_name
                        _state["pending"] = None
                        _state["count"] = 0
                else:
                    _state["pending"] = gmm_regime_name
                    _state["count"] = 1

                smoothed_regime = _state["current"]
                if smoothed_regime != gmm_regime_name:
                    logger.info(
                        f"[REGIME_SMOOTH] {asset}: {gmm_regime_name} suppressed -> "
                        f"holding {smoothed_regime} (pending={_state['pending']},"
                        f"count={_state['count']}/{self._regime_smoother_persistence})"
                    )
                gmm_regime_name = smoothed_regime
                raw["regime_state"] = smoothed_regime
            raw["_regime_raw"] = _raw_regime

            # ─── BEST-OF-N STRATEGY ─────────────────────────────────────
            strategies = self._compute_strategy_signals(
                rsi_sig, macd_sig, bb_sig, ema_sig, adx_mult,
                _rsi, _adx, bb_pos if bb_rng > 0 else 0.0,
                raw.get("volume_ratio", 1.0), gmm_regime_name,
            )

            # [SENT-SWITCH] Sentiment-driven strategy fitness modulation
            # Extreme fear → boost mean_revert (contrarian), suppress momentum
            # Extreme greed → boost momentum (trend-following), suppress mean_revert
            # Neutral → no modification (sentiment stays silent)
            _fg_value = raw.get("_fear_greed_value", 50)
            if _fg_value is not None and isinstance(_fg_value, (int, float)):
                if _fg_value < 25:  # Extreme fear
                    _sent_boost = (25 - _fg_value) / 25.0  # 0→1.0 as fear increases
                    if "mean_revert" in strategies:
                        strategies["mean_revert"]["strength"] *= (1.0 + 0.4 * _sent_boost)
                    if "momentum" in strategies:
                        strategies["momentum"]["strength"] *= (1.0 - 0.3 * _sent_boost)
                elif _fg_value > 75:  # Extreme greed
                    _sent_boost = (_fg_value - 75) / 25.0  # 0→1.0 as greed increases
                    if "momentum" in strategies:
                        strategies["momentum"]["strength"] *= (1.0 + 0.3 * _sent_boost)
                    if "mean_revert" in strategies:
                        strategies["mean_revert"]["strength"] *= (1.0 - 0.2 * _sent_boost)
                if _fg_value < 20 or _fg_value > 80:
                    if "vrp" in strategies:
                        strategies["vrp"]["strength"] *= 1.3  # Extreme sentiment → vol premium

            # [FIX-MOM-CONSOLIDATION] Hard filter: momentum must not win in
            # low-vol consolidation regimes. Paper run showed momentum in
            # QUIET_ACCUMULATION/WEAK_CONSOLIDATION lost on every exit.
            # Cap momentum strength to 0.10 so mean_revert (typ 0.13-0.14)
            # wins whenever it has a signal. Was 0.15 but momentum still
            # won by 0.01 margin due to EMA_200 systematic short bias.
            _MOM_SUPPRESSED_REGIMES = {"QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"}
            if gmm_regime_name in _MOM_SUPPRESSED_REGIMES and "momentum" in strategies:
                strategies["momentum"]["strength"] = min(
                    strategies["momentum"]["strength"], 0.10
                )

            best_name = max(strategies, key=lambda k: strategies[k]["strength"])
            best = strategies[best_name]
            quant_dir = float(np.clip(best["signal"], -1, 1))

            signs = [1 if c > 0 else (-1 if c < 0 else 0)
                     for c in [rsi_sig, macd_sig, bb_sig, ema_sig]]
            agreement = abs(sum(signs)) / 4.0
            quant_conf = 0.3 + 0.4 * agreement + 0.3 * min(abs(quant_dir), 1.0)

            # [P3-LONG-BIAS] In consolidation regimes, add slight LONG bias.
            # Paper run evidence: ALL profitable trades were LONG, ALL shorts lost.
            # Crypto in QUIET_ACCUM tends to drift up (accumulation before breakout).
            # Bias is small (+0.10) — not enough to override strong SHORT signals,
            # but enough to tip neutral/weak signals toward LONG.
            _LONG_BIAS_REGIMES = {"QUIET_ACCUMULATION", "WEAK_CONSOLIDATION"}
            if gmm_regime_name in _LONG_BIAS_REGIMES:
                quant_dir = float(np.clip(quant_dir + 0.10, -1.0, 1.0))

            raw["quant_direction"] = quant_dir
            raw["quant_confidence"] = quant_conf
            raw["strategy_agreement"] = agreement
            # [FIX-ALPHA-CAL] Was 150 → realized alpha analysis shows +41bps avg win
            # but 50% win rate → effective ~20bps. With avg |quant_dir|≈0.3:
            # multiplier = 20 / 0.3 ≈ 65. Use 65 to align estimate with reality.
            raw["signal_edge_bps"] = abs(quant_dir) * 65
            raw["primary_strategy"] = best_name
            raw["_strategy_selection"] = {
                "best": best_name,
                "best_signal": best["signal"],
                "best_strength": best["strength"],
                "best_direction": "LONG" if best["signal"] > 0 else "SHORT" if best["signal"] < 0 else "FLAT",
                "regime": gmm_regime_name,
                "all": {k: {"signal": v["signal"], "strength": v["strength"]}
                        for k, v in strategies.items()},
            }

            other_strs = ', '.join(
                f'{k}={v["strength"]:.2f}' for k, v in strategies.items() if k != best_name
            )
            logger.info(
                f"[STRATEGY_SELECT] {asset}: Best={best_name} "
                f"(sig={best['signal']:+.3f}, str={best['strength']:.3f}) | "
                f"regime={gmm_regime_name} | others=[{other_strs}]"
            )

            if self._sq_tracker:
                try:
                    self._sq_tracker.record_candidates(
                        asset=asset,
                        candidates={k: {"signal": v["signal"], "strength": v["strength"]}
                                    for k, v in strategies.items()},
                        selected=best_name,
                        regime=gmm_regime_name or "UNKNOWN",
                        tick=tick_count,
                    )
                except Exception as _sq_err:
                    logger.debug(f"[SQ] {asset}: record_candidates skipped: {_sq_err}")

            # ─── RELATIVE ALPHA ─────────────────────────────────────────
            _close_vals = df["close"].values[-25:] if len(df) >= 25 else df["close"].values
            _denom = np.where(_close_vals[:-1] != 0, _close_vals[:-1], 1.0)
            rets_4h = np.diff(_close_vals) / _denom if len(_close_vals) >= 2 else np.array([0.0])
            self.asset_returns_cache[asset] = {
                "rets": rets_4h,
                "prices": _close_vals.tolist() if hasattr(_close_vals, 'tolist') else list(_close_vals),
                "timestamp": _time.time(),
            }

            _CACHE_MAX_AGE_SEC = 300
            btc_cache = self.asset_returns_cache.get("BTC")
            if asset != "BTC" and btc_cache is not None:
                cache_age = _time.time() - btc_cache["timestamp"]
                if cache_age > _CACHE_MAX_AGE_SEC:
                    btc_cache = None

            if asset != "BTC" and btc_cache is not None:
                # L4-16: Warn on cross-asset cache skew
                _skew_sec = abs(_time.time() - btc_cache["timestamp"])
                if _skew_sec > 60:
                    logger.debug(
                        f"[CACHE_SKEW] {asset}-BTC returns cache offset: {_skew_sec:.0f}s"
                    )
                btc_rets = btc_cache["rets"]
                min_len = min(len(btc_rets), len(rets_4h))
                if min_len >= 10:
                    _cov = np.cov(rets_4h[-min_len:], btc_rets[-min_len:])
                    _beta = _cov[0, 1] / _cov[1, 1] if _cov.ndim == 2 and _cov[1, 1] > 1e-15 else 1.0
                    rel_alpha = float(rets_4h[-1] - _beta * btc_rets[-1])
                    if not np.isfinite(rel_alpha):
                        rel_alpha = 0.0
                    raw["relative_alpha_vs_btc"] = rel_alpha
                    raw["beta_to_btc"] = float(_beta)
                    if rel_alpha < -0.02:
                        raw["relative_strength_signal"] = -1
                    elif rel_alpha > 0.02:
                        raw["relative_strength_signal"] = 1
                    else:
                        raw["relative_strength_signal"] = 0
                else:
                    raw["relative_alpha_vs_btc"] = 0.0
                    raw["beta_to_btc"] = 1.0
                    raw["relative_strength_signal"] = 0
            else:
                raw["relative_alpha_vs_btc"] = 0.0
                raw["beta_to_btc"] = 1.0
                raw["relative_strength_signal"] = 0

            # Feed intraday correlation monitor
            if self._correlation_monitor:
                try:
                    _btc_cache = self.asset_returns_cache.get("BTC", {})
                    _eth_cache = self.asset_returns_cache.get("ETH", {})
                    _sol_cache = self.asset_returns_cache.get("SOL", {})
                    _btc_p = _btc_cache.get("prices", [0])[-1] if _btc_cache.get("prices") else 0
                    _eth_p = _eth_cache.get("prices", [0])[-1] if _eth_cache.get("prices") else 0
                    _sol_p = _sol_cache.get("prices", [0])[-1] if _sol_cache.get("prices") else 0
                    if _btc_p > 0 and _eth_p > 0 and _sol_p > 0:
                        self._correlation_monitor.update_prices(_btc_p, _eth_p, _sol_p)
                except Exception as e:
                    logger.debug(f"[CORRELATION] Price feed error: {e}")

            # Cross-asset correlation
            _btc_c = self.asset_returns_cache.get("BTC")
            _eth_c = self.asset_returns_cache.get("ETH")
            _sol_c = self.asset_returns_cache.get("SOL")
            if _btc_c and _eth_c and _sol_c:
                _b = _btc_c["rets"][-20:]
                _e = _eth_c["rets"][-20:]
                _s = _sol_c["rets"][-20:]
                _min_n = min(len(_b), len(_e), len(_s))
                if _min_n >= 10:
                    _cb = _b[-_min_n:]
                    _ce = _e[-_min_n:]
                    _cs = _s[-_min_n:]
                    _c_be = float(np.corrcoef(_cb, _ce)[0, 1]) if np.std(_cb) > 0 and np.std(_ce) > 0 else 0.85
                    _c_bs = float(np.corrcoef(_cb, _cs)[0, 1]) if np.std(_cb) > 0 and np.std(_cs) > 0 else 0.80
                    _c_es = float(np.corrcoef(_ce, _cs)[0, 1]) if np.std(_ce) > 0 and np.std(_cs) > 0 else 0.80
                    _xcorr = float(np.clip((_c_be + _c_bs + _c_es) / 3.0, -1.0, 1.0))
                    if np.isfinite(_xcorr):
                        raw["cross_asset_correlation"] = _xcorr

            # ─── HIGHER LOWS ────────────────────────────────────────────
            _lows = df["low"].values
            _hl_window = 5
            _local_mins = []
            for _i in range(_hl_window, len(_lows) - _hl_window):
                if _lows[_i] == min(_lows[_i - _hl_window: _i + _hl_window + 1]):
                    _local_mins.append(float(_lows[_i]))
            raw["higher_lows_detected"] = False
            raw["higher_lows_strength"] = 0.0
            if len(_local_mins) >= 3:
                _recent = _local_mins[-3:]
                _is_hl = all(_recent[j] > _recent[j - 1] for j in range(1, len(_recent)))
                raw["higher_lows_detected"] = _is_hl
                if _is_hl and len(_local_mins) >= 2:
                    _hl_gains = [
                        (_local_mins[j] - _local_mins[j - 1]) / _local_mins[j - 1]
                        for j in range(1, len(_local_mins))
                        if _local_mins[j - 1] > 0 and _local_mins[j] > _local_mins[j - 1]
                    ]
                    raw["higher_lows_strength"] = min(1.0, sum(_hl_gains) / max(len(_hl_gains), 1) * 50)

            _raw_vol_ratio_log = float(raw.get("volume_ratio", 1.0) or 0.0)
            _effective_vol_ratio_log = float(
                raw.get("volume_ratio_effective", _raw_vol_ratio_log) or _raw_vol_ratio_log
            )
            _vol_ratio_text = f"{_raw_vol_ratio_log:.2f}"
            if abs(_effective_vol_ratio_log - _raw_vol_ratio_log) >= 0.05:
                _vol_ratio_text = (
                    f"{_raw_vol_ratio_log:.2f} eff={_effective_vol_ratio_log:.2f}"
                )

            logger.info(
                f"[QUANT_SIGNAL] {asset}: dir={quant_dir:+.3f} | conf={quant_conf:.2f} | "
                f"regime_conf={regime_conf:.2f} "
                f"{'[GMM:' + gmm_regime_name + ']' if gmm_regime_name else '[ADX]'} | "
                f"rel_alpha={raw.get('relative_alpha_vs_btc', 0):.4f} | "
                f"HL={raw.get('higher_lows_detected', False)} | "
                f"vol_ratio={_vol_ratio_text} | "
                f"[rsi={rsi_sig:+.2f} macd={macd_sig:+.2f} "
                f"bb={bb_sig:+.2f} ema={ema_sig:+.2f}] x adx_mult={adx_mult:.2f}"
            )

        except Exception as e:
            logger.error(f"[PREPARE_DATA] {asset}: indicator calculation failed: {e}")

        # ─── Post-processing (Section A) ────────────────────────────────
        depth_1pct = raw.get('orderbook_depth_1pct_usd', 500000)
        _asset_key = asset.upper().replace("/USD", "")
        _depth_buf = self._depth_history.get(_asset_key)
        _depth_reliable = (
            (not raw.get("orderbook_stale", False))
            or raw.get("orderbook_fallback_reason") == "cached_depth"
        )
        if _depth_buf is not None and _depth_reliable and depth_1pct > 0 and depth_1pct < float('inf'):
            _depth_buf.append(depth_1pct)
        if _depth_buf and len(_depth_buf) >= 3:
            import statistics as _stats_mod
            _depth_median = _stats_mod.median(_depth_buf)
        else:
            _depth_median = depth_1pct if depth_1pct > 0 else 500_000
        _raw_depth = min(depth_1pct / _depth_median, 5.0) if (_depth_median > 0 and depth_1pct < float('inf')) else 1.0
        raw['orderbook_depth'] = _raw_depth if (not math.isnan(_raw_depth) and not math.isinf(_raw_depth)) else 1.0

        _spread = raw.get('spread_bps', 10.0)
        _impact_bps = min(20.0, 50_000.0 / max(depth_1pct, 1.0) * 100) if depth_1pct > 0 else 5.0
        raw['estimated_friction_bps'] = _spread + _impact_bps

        _prices = raw.get('prices', [])
        if len(_prices) >= 2 and _prices[-2] > 0:
            raw['price_move_pct'] = (_prices[-1] - _prices[-2]) / _prices[-2]
        else:
            raw['price_move_pct'] = 0.0

        if len(_prices) >= 2 and _prices[-2] > 0:
            raw['price_change_4h_pct'] = (_prices[-1] - _prices[-2]) / _prices[-2]
        else:
            raw['price_change_4h_pct'] = 0.0
        raw['price_change_1h_pct'] = raw['price_change_4h_pct'] / 4.0

        _fast_move = abs(raw.get('price_move_pct', 0)) > 0.05
        _depth_collapsed = _depth_reliable and (
            _depth_buf is not None
            and len(_depth_buf) >= 5
            and depth_1pct < _depth_median * 0.5
        ) if _depth_median > 0 else False
        raw['flash_crash_active'] = _fast_move or (
            _fast_move is False and _depth_collapsed
            and abs(raw.get('price_move_pct', 0)) > 0.02
        )

        if 'dvol_zscore' not in raw and 'dvol' in raw:
            raw['dvol_zscore'] = raw['dvol']

        _REQUIRED_V36_KEYS = [
            'orderbook_depth', 'estimated_friction_bps', 'spread_bps',
            'current_price', 'dvol_zscore', 'vpin',
        ]
        _missing = [k for k in _REQUIRED_V36_KEYS if k not in raw]
        if _missing:
            logger.warning(f"[MARKET_DATA] {asset}: missing v36 keys: {_missing}")
            raw['_v36_data_degraded'] = True
        else:
            raw['_v36_data_degraded'] = False

        # [FIX-41] NaN/inf sanitizer
        _nan_count = 0
        for _k, _v in raw.items():
            if isinstance(_v, float) and not math.isfinite(_v):
                _nan_count += 1
                raw[_k] = 0.0
        if _nan_count > 0:
            logger.warning(f"[FIX-41] {asset}: sanitized {_nan_count} NaN/inf values")

        return raw

    # ─── Private methods ────────────────────────────────────────────────

    async def _fetch_live_data(self, asset: str) -> Dict[str, Any]:
        """Fetch live market data from Kraken via CCXT/REST."""
        symbol_map = {"SOL": "SOL/USD", "BTC": "BTC/USD", "ETH": "ETH/USD"}
        symbol = symbol_map.get(asset, f"{asset}/USD")
        t0 = _time.monotonic()
        target_feature_bars = 1024
        live_bar_limit = 721

        try:
            exchange = None
            if self._kraken_rest and self._kraken_rest.is_initialized():
                exchange = self._kraken_rest._exchange
            elif self._ccxt_exchange:
                exchange = self._ccxt_exchange

            if exchange is None:
                raise RuntimeError("No CCXT exchange available")

            # [P-7] Parallel CCXT fetch
            loop = asyncio.get_running_loop()
            ticker_fut = loop.run_in_executor(
                self._sync_executor, exchange.fetch_ticker, symbol)
            ohlcv_fut = loop.run_in_executor(
                self._sync_executor, exchange.fetch_ohlcv, symbol, "4h", None, live_bar_limit)
            ob_fut = loop.run_in_executor(
                self._sync_executor, exchange.fetch_order_book, symbol, 50)
            # [OOD-CAL] Always fetch recent trades so runtime external features
            # can mirror training trade activity features.
            trades_fut = loop.run_in_executor(
                self._sync_executor, exchange.fetch_trades, symbol, None, 1000
            )
            _gather_futs = [ticker_fut, ohlcv_fut, ob_fut]
            if trades_fut:
                _gather_futs.append(trades_fut)
            _gather_res = await asyncio.gather(*_gather_futs, return_exceptions=True)
            ticker_res, ohlcv_res, ob_res = _gather_res[0], _gather_res[1], _gather_res[2]
            _trades_res = _gather_res[3] if len(_gather_res) > 3 else None

            if isinstance(ticker_res, BaseException):
                raise ticker_res
            ticker = ticker_res
            current_price = float(ticker.get("last", 0))
            volume_24h = float(ticker.get("quoteVolume", 0))
            bid = float(ticker.get("bid", current_price))
            ask = float(ticker.get("ask", current_price))
            spread_bps = ((ask - bid) / current_price * 10000) if current_price > 0 else 0.0
            exchange_ts = ticker.get("timestamp", None)

            if isinstance(ohlcv_res, BaseException):
                raise ohlcv_res
            ohlcv = ohlcv_res
            ohlcv_raw = []
            prices = []
            volumes = []
            base_volumes = []
            for bar in ohlcv:
                _close = float(bar[4])
                _base_volume = float(bar[5])
                _quote_volume = _base_volume * _close
                ohlcv_raw.append([bar[0], bar[1], bar[2], bar[3], _close, _quote_volume])
                prices.append(_close)
                volumes.append(_quote_volume)
                base_volumes.append(_base_volume)

            if ohlcv_raw:
                bootstrap = self._get_bootstrap_ohlcv(
                    asset,
                    before_ts_ms=int(ohlcv_raw[0][0]),
                    needed_bars=max(0, target_feature_bars - len(ohlcv_raw)),
                )
                if bootstrap:
                    ohlcv_raw = bootstrap + ohlcv_raw
                    prices = [float(bar[4]) for bar in ohlcv_raw]
                    volumes = [float(bar[5]) for bar in ohlcv_raw]
                    base_volumes = ([0.0] * len(bootstrap)) + base_volumes

            drl_ohlcv_raw = ohlcv_raw
            if ohlcv_raw:
                try:
                    drl_ohlcv_raw = self._build_drl_ohlcv(asset, ohlcv_raw, target_feature_bars)
                except Exception as _drl_hist_err:
                    logger.debug(f"[DRL_WARMUP] {asset}: DRL history build failed: {_drl_hist_err}")
                    drl_ohlcv_raw = ohlcv_raw

            orderbook_depth_1pct = 500_000.0
            order_book_imbalance = 0.0
            orderbook_liquidity_score = 1.0
            orderbook_slippage_buy = 0.0
            orderbook_slippage_sell = 0.0
            orderbook_stale = False
            orderbook_fallback_reason = "live"
            orderbook_cache_age_seconds = 0.0
            orderbook_failure_streak = 0
            try:
                if isinstance(ob_res, BaseException):
                    raise ob_res
                ob = ob_res
                if not (ob and ob.get("bids") and ob.get("asks")):
                    raise RuntimeError("empty orderbook")
                mid = (ob["bids"][0][0] + ob["asks"][0][0]) / 2
                threshold = mid * 0.01
                bid_depth = sum(
                    lvl[0] * lvl[1] for lvl in ob["bids"] if lvl[0] >= mid - threshold
                )
                ask_depth = sum(
                    lvl[0] * lvl[1] for lvl in ob["asks"] if lvl[0] <= mid + threshold
                )
                orderbook_depth_1pct = bid_depth + ask_depth
                _ob_total = bid_depth + ask_depth
                order_book_imbalance = (bid_depth - ask_depth) / _ob_total if _ob_total > 0 else 0.0

                if self._orderbook_analyzer:
                    try:
                        self._orderbook_analyzer.update_order_book(
                            symbol=symbol,
                            bids=[(float(lvl[0]), float(lvl[1])) for lvl in ob["bids"]],
                            asks=[(float(lvl[0]), float(lvl[1])) for lvl in ob["asks"]],
                        )
                        _ob_analysis = self._orderbook_analyzer.analyze(symbol, order_size_usd=10000)
                        if _ob_analysis:
                            orderbook_liquidity_score = _ob_analysis.liquidity_score
                            orderbook_slippage_buy = _ob_analysis.estimated_slippage_buy
                            orderbook_slippage_sell = _ob_analysis.estimated_slippage_sell
                    except Exception as _ob_err:
                        logger.debug(f"[ORDERBOOK] {asset}: analysis failed: {_ob_err}")

                _prev_failures = self._orderbook_failure_streak.get(asset, 0)
                if _prev_failures > 0:
                    logger.info(f"[LIVE_DATA] {asset}: Orderbook recovered after {_prev_failures} failures")
                self._orderbook_failure_streak[asset] = 0
                self._last_orderbook_depth_usd[asset] = float(orderbook_depth_1pct)
                self._last_orderbook_imbalance[asset] = float(order_book_imbalance)
                self._last_orderbook_ts[asset] = _time.time()
            except Exception as _ob_err:
                _streak = self._orderbook_failure_streak.get(asset, 0) + 1
                self._orderbook_failure_streak[asset] = _streak
                orderbook_failure_streak = _streak
                orderbook_stale = True
                _cached_depth = self._last_orderbook_depth_usd.get(asset)
                _cached_imbalance = self._last_orderbook_imbalance.get(asset, 0.0)
                _cached_ts = self._last_orderbook_ts.get(asset, 0.0)
                _cache_age = (_time.time() - _cached_ts) if _cached_ts > 0 else float("inf")

                if _cached_depth is not None and _cache_age <= self._orderbook_stale_ttl_seconds:
                    orderbook_depth_1pct = float(_cached_depth)
                    order_book_imbalance = float(_cached_imbalance)
                    orderbook_fallback_reason = "cached_depth"
                    orderbook_cache_age_seconds = float(_cache_age)
                    _ob_msg = (
                        f"[LIVE_DATA] {asset}: Orderbook fetch failed: {_ob_err}; "
                        f"using cached depth (${orderbook_depth_1pct:,.0f}, age={_cache_age:.1f}s, streak={_streak})"
                    )
                else:
                    orderbook_fallback_reason = "static_floor"
                    orderbook_cache_age_seconds = None
                    _ob_msg = (
                        f"[LIVE_DATA] {asset}: Orderbook fetch failed: {_ob_err}; "
                        f"using static floor depth (streak={_streak})"
                    )

                # Reduce noise: warn on first failure and periodic checkpoints only.
                if _streak == 1 or _streak % 10 == 0:
                    logger.warning(_ob_msg)
                else:
                    logger.debug(_ob_msg)

            # OFI z-score: rolling z-score of order_book_imbalance
            _ofi_zscore = 0.0
            if asset in self._ofi_history:
                self._ofi_history[asset].append(order_book_imbalance)
                _ofi_buf = self._ofi_history[asset]
                if len(_ofi_buf) >= 5:
                    _ofi_arr = np.array(_ofi_buf)
                    _ofi_mean = np.mean(_ofi_arr)
                    _ofi_std = np.std(_ofi_arr)
                    if _ofi_std > 1e-8:
                        _ofi_zscore = float(np.clip(
                            (order_book_imbalance - _ofi_mean) / _ofi_std, -5.0, 5.0
                        ))

            returns_4h = 0.0
            if len(prices) >= 2 and prices[-2] > 0:
                returns_4h = (prices[-1] - prices[-2]) / prices[-2]

            volatility_4h = 0.02
            if len(prices) >= 20:
                import statistics
                rets = [(prices[i] - prices[i-1]) / prices[i-1]
                        for i in range(-19, 0) if prices[i-1] > 0]
                if len(rets) > 1:
                    volatility_4h = statistics.stdev(rets)

            fetch_latency = _time.monotonic() - t0

            if exchange_ts is not None:
                data_age = max(0.0, _time.time() - exchange_ts / 1000.0)
            else:
                data_age = 0.5

            _ob_depth_str = f"${orderbook_depth_1pct:,.0f}" if orderbook_depth_1pct < float('inf') else "UNAVAIL"
            logger.info(
                f"[LIVE_DATA] {asset}: ${current_price:,.2f} | "
                f"vol24h=${volume_24h:,.0f} | spread={spread_bps:.1f}bps | "
                f"depth_1pct={_ob_depth_str} | "
                f"bars={len(prices)} | age={data_age:.2f}s | fetch={fetch_latency:.2f}s"
            )

            # [VPIN-B] Compute VPIN from fetched trades
            _vpin_val = 0.35
            _vpin_source = "synthetic"
            if self._vpin_calc and _trades_res is not None:
                if not isinstance(_trades_res, BaseException):
                    try:
                        _parsed_trades = [
                            {"price": float(t.get("price", 0)),
                             "volume": float(t.get("amount", 0))}
                            for t in _trades_res
                        ]
                        _v = self._vpin_calc.compute_from_trades(_parsed_trades, asset)
                        if _v >= 0:
                            _vpin_val = _v
                            _vpin_source = "computed"
                    except Exception as _ve:
                        logger.debug(f"[VPIN] {asset}: compute failed: {_ve}")
                else:
                    logger.debug(f"[VPIN] {asset}: fetch_trades failed: {_trades_res}")

            # [SIG-1] Whale detection via WhaleDetector (3-dim: ABSOLUTE + RELATIVE_SIZE + DEPTH_IMPACT)
            _whale_dir = "none"
            _whale_buy_vol = 0.0
            _whale_sell_vol = 0.0
            _whale_net_pressure = 0.0
            _whale_count = 0
            if _WHALE_DETECTOR_AVAILABLE and _trades_res is not None and not isinstance(_trades_res, BaseException):
                try:
                    _wd = get_whale_detector()
                    for _wt in _trades_res:
                        _wt_price = float(_wt.get("price", 0))
                        _wt_amount = float(_wt.get("amount", 0))
                        _wt_notional = _wt_price * _wt_amount
                        _wt_side = str(_wt.get("side", "")).upper()
                        if _wt_side in ("BUY", "SELL") and _wt_notional > 0:
                            _wd.detect(asset, _wt_notional, _wt_side, orderbook_depth_1pct)
                    _wp = _wd.get_pressure(asset)
                    _whale_buy_vol = _wp.buy_volume_usd
                    _whale_sell_vol = _wp.sell_volume_usd
                    _whale_net_pressure = _wp.net_pressure
                    _whale_count = _wp.whale_count
                    if _wp.net_pressure > 0.3:
                        _whale_dir = "buy"
                    elif _wp.net_pressure < -0.3:
                        _whale_dir = "sell"
                    elif _whale_count > 0:
                        _whale_dir = "mixed"
                except Exception as _wd_err:
                    logger.debug(f"[SIG-1] {asset}: whale detection skipped: {_wd_err}")  # fail-open

            _trade_count = 0.0
            _trade_notional_usd = 0.0
            _trade_buy_notional_usd = 0.0
            _trade_sell_notional_usd = 0.0
            _trade_metrics_source = "none"
            if _trades_res is not None and not isinstance(_trades_res, BaseException):
                try:
                    for _tt in _trades_res:
                        _price = self._safe_float(_tt.get("price", 0.0))
                        _amount = self._safe_float(_tt.get("amount", 0.0))
                        _side = str(_tt.get("side", "")).lower()
                        _notional = _price * _amount
                        if _notional <= 0:
                            continue
                        _trade_notional_usd += _notional
                        if _side == "buy":
                            _trade_buy_notional_usd += _notional
                        elif _side == "sell":
                            _trade_sell_notional_usd += _notional
                    _trade_count = float(len(_trades_res))
                    _trade_metrics_source = "kraken_trades"
                except Exception as _tm_err:
                    logger.debug(f"[LIVE_DATA] {asset}: trade metric parse failed: {_tm_err}")

            if _trade_notional_usd <= 0.0 and ohlcv:
                try:
                    _last_bar_volume = self._safe_float(ohlcv[-1][5], 0.0) if len(ohlcv[-1]) > 5 else 0.0
                    _trade_notional_usd = max(_trade_notional_usd, _last_bar_volume * current_price)
                    if _trade_count <= 0.0 and _trade_notional_usd > 0.0:
                        _avg_ticket = max(current_price * 0.02, 50.0)
                        _trade_count = float(max(1, min(5000, int(_trade_notional_usd / _avg_ticket))))
                    if _trade_metrics_source == "none":
                        _trade_metrics_source = "ohlcv_proxy"
                except Exception:
                    pass

            # [FIX-C1] Cross-asset correlation: computed in fetch_and_prepare() (line ~1060)
            # and injected into this dict post-return. Use GMM training default 0.87 here;
            # fetch_and_prepare() will overwrite with the real value.
            _cross_asset_corr = 0.87

            _fetch_completed_ts = _time.time()
            _latest_bar_open_ts_ms = int(ohlcv_raw[-1][0]) if ohlcv_raw else 0

            return {
                "current_price": current_price,
                "prices": prices, "volumes": volumes, "ohlcv_raw": ohlcv_raw,
                "drl_ohlcv_raw": drl_ohlcv_raw,
                "base_volumes": base_volumes,
                "volume_24h": volume_24h, "spread_bps": spread_bps,
                "returns_4h": returns_4h, "volatility_4h": volatility_4h,
                "data_age_seconds": data_age,
                "data_age_sec": data_age,
                "fetch_latency_seconds": fetch_latency,
                "market_data_fetched_at_ts": _fetch_completed_ts,
                "latest_bar_open_ts_ms": _latest_bar_open_ts_ms,
                "data_valid": True,
                "dvol_zscore": 0.0, "vpin": _vpin_val, "vpin_source": _vpin_source, "ofi_zscore": _ofi_zscore,
                "correlation_btc_eth_sol": _cross_asset_corr,
                "orderbook_depth_1pct_usd": orderbook_depth_1pct,
                "order_book_imbalance": order_book_imbalance,
                "orderbook_imbalance": order_book_imbalance,
                "orderbook_stale": orderbook_stale,
                "orderbook_fallback_reason": orderbook_fallback_reason,
                "orderbook_cache_age_seconds": orderbook_cache_age_seconds,
                "orderbook_failure_streak": orderbook_failure_streak,
                "orderbook_liquidity_score": orderbook_liquidity_score,
                "orderbook_slippage_buy_bps": orderbook_slippage_buy,
                "orderbook_slippage_sell_bps": orderbook_slippage_sell,
                "current_exposure": self._paper_positions.get(asset, {}).get("exposure", 0.0),
                "position_direction": int(self._paper_positions.get(asset, {}).get("direction", 0)),
                "current_tranche": int(self._paper_positions.get(asset, {}).get("tranche", 0)),
                "current_drawdown": self.current_drawdown_pct,
                "cross_asset_correlation": _cross_asset_corr,
                "sol_exposure": self._paper_positions.get("SOL", {}).get("exposure", 0.0),
                "btc_exposure": self._paper_positions.get("BTC", {}).get("exposure", 0.0),
                "eth_exposure": self._paper_positions.get("ETH", {}).get("exposure", 0.0),
                "no_trade_triggers": {},
                "is_weekend": self._weekend_manager.update().is_weekend if self._weekend_manager else self._is_weekend_now(),
                "is_holiday": False, "is_low_liquidity": False,
                "asset": asset, "_source": "kraken_live",
                "whale_direction": _whale_dir,
                "whale_buy_volume_usd": _whale_buy_vol,
                "whale_sell_volume_usd": _whale_sell_vol,
                "whale_net_pressure": _whale_net_pressure,
                "whale_count": _whale_count,
                "recent_trade_count": _trade_count,
                "recent_taker_volume_usd": _trade_notional_usd,
                "recent_buy_trade_volume_usd": _trade_buy_notional_usd,
                "recent_sell_trade_volume_usd": _trade_sell_notional_usd,
                "trade_metrics_source": _trade_metrics_source,
            }
            # [2026-04-10] Reset global failure counter on successful fetch
            self._global_fetch_failure_count = 0
            self._global_fetch_success_count += 1

        except Exception as e:
            logger.warning(f"[LIVE_DATA] {asset} fetch failed: {e}, using synthetic fallback")
            self._orderbook_failure_streak[asset] = self._orderbook_failure_streak.get(asset, 0) + 1
            # [2026-04-10] CCXT session auto-reset on consecutive failures
            self._global_fetch_failure_count += 1
            if self._global_fetch_failure_count >= self._CCXT_RESET_THRESHOLD:
                self._reset_ccxt_session()
            fallback = self.generate_verification_data(asset)
            fallback["data_valid"] = False
            fallback["_source"] = "synthetic_fallback"
            fallback["orderbook_stale"] = True
            fallback["orderbook_fallback_reason"] = "synthetic_fallback"
            fallback["orderbook_cache_age_seconds"] = None
            fallback["orderbook_failure_streak"] = self._orderbook_failure_streak.get(asset, 1)
            return fallback

    def _predict_gmm_regime(self, df, raw: Dict, _np) -> Optional[tuple]:
        """Predict regime using pretrained 12-feature GMM model (v3)."""
        closes = df["close"].values
        vols = df["volume"].values
        n = len(closes)
        if n < 50:
            return None

        asset = raw.get("asset", "?")

        # [Phase 3] GMM cache
        _gmm_cache_key = f"{asset}_gmm"
        _last_bar_ts = closes[-1] if n > 0 else 0
        _ta_c = self._ta_cache.get(asset)
        if _ta_c:
            _last_bar_ts = _ta_c.get("last_ts", _last_bar_ts)
        _gmm_cached = self._ta_cache.get(_gmm_cache_key)
        if (_gmm_cached is not None
                and _gmm_cached["ts"] == _last_bar_ts
                and _gmm_cached["n"] == n):
            for _gk, _gv in _gmm_cached.get("raw_keys", {}).items():
                raw[_gk] = _gv
            return _gmm_cached["result"]

        rets = _np.diff(closes) / _np.where(closes[:-1] != 0, closes[:-1], 1.0)

        ret_4h = float(rets[-1]) if len(rets) > 0 else 0.0
        ret_1h = ret_4h / 4.0
        ret_24h = float((closes[-1] - closes[-7]) / closes[-7]) if n >= 7 and closes[-7] > 0 else 0.0
        ret_7d = float((closes[-1] - closes[-43]) / closes[-43]) if n >= 43 and closes[-43] > 0 else 0.0

        vol_1h = float(_np.std(rets[-2:])) if len(rets) >= 2 else 0.02
        vol_24h = float(_np.std(rets[-6:])) if len(rets) >= 6 else 0.02
        vol_pct = float(_np.searchsorted(_np.sort(vols), vols[-1]) / n * 100) if n > 0 else 50.0

        if len(rets) >= 30:
            _wv = _np.lib.stride_tricks.sliding_window_view(rets, 6)
            rolling_v = _np.std(_wv, axis=1)
            vov = float(_np.std(rolling_v[-20:])) if len(rolling_v) >= 20 else 0.0
        else:
            vov = 0.0

        if len(rets) >= 6:
            recent = rets[-6:]
            pos = int(_np.sum(recent > 0))
            neg = int(_np.sum(recent < 0))
            mom_con = (max(pos, neg) / 6.0) * (1 if pos >= neg else -1)
        else:
            mom_con = 0.0

        cross_corr = raw.get("cross_asset_correlation", 0.87)
        fear_idx = 100.0 - raw.get("rsi", 50.0)
        _spread_defaults = {"BTC": 5.0, "ETH": 8.0, "SOL": 12.0}
        spread_pct = _spread_defaults.get(asset, 8.0)

        if asset in self._gmm_configs:
            _cfg = self._gmm_configs[asset]
            _model = self._gmm_models[asset]
            _scaler_mean = _cfg["scaler_mean"]
            _scaler_scale = _cfg["scaler_scale"]
            _regime_names = _cfg["regime_names"]
            feature_names = _cfg["feature_cols"]
        else:
            _model = self._gmm_model
            _scaler_mean = self._gmm_scaler_mean
            _scaler_scale = self._gmm_scaler_scale
            _regime_names = self._gmm_regime_names
            feature_names = self._gmm_feature_cols

        features = _np.array([
            ret_1h, ret_4h, ret_24h, ret_7d,
            vol_1h, vol_24h, vol_pct, vov,
            mom_con, cross_corr, fear_idx, spread_pct,
        ], dtype=_np.float64)

        if _scaler_mean is None or _scaler_scale is None:
            logger.warning(f"[GMM] {asset}: scaler not loaded, falling back to ADX proxy")
            raw["_gmm_fallback"] = "scaler_missing"
            return None
        scaled = (features - _scaler_mean) / _scaler_scale

        extreme_mask = _np.abs(scaled) > 3.0
        n_extreme = int(_np.sum(extreme_mask))
        if n_extreme > 0:
            extreme_details = []
            for i in range(len(scaled)):
                if extreme_mask[i]:
                    fname = feature_names[i] if i < len(feature_names) else f"f{i}"
                    extreme_details.append(f"{fname}(raw={features[i]:.4f},z={scaled[i]:.2f})")
            logger.warning(
                f"[GMM_DIAG] {asset}: {n_extreme}/{len(scaled)} features |z|>3: "
                f"{', '.join(extreme_details)}"
            )

        if n_extreme > len(scaled) * 0.3:
            logger.warning(
                f"[GMM_DIAG] {asset}: distribution shift detected, falling back to ADX proxy"
            )
            raw["_gmm_fallback"] = "distribution_shift"
            return None

        scaled = _np.clip(scaled, -4.0, 4.0)

        probs = _model.predict_proba(scaled.reshape(1, -1))[0]
        regime_idx = int(_np.argmax(probs))
        confidence = float(probs[regime_idx])
        regime_name = (
            _regime_names[regime_idx]
            if regime_idx < len(_regime_names)
            else f"R{regime_idx}"
        )

        if confidence > 0.999:
            logger.info(
                f"[GMM] {asset}: conf={confidence:.8f} for {regime_name} "
                f"(probs={[f'{p:.2e}' for p in probs]}) - high but z-scores normal, using GMM"
            )

        raw["_gmm_regime_idx"] = regime_idx
        raw["_gmm_regime_name"] = regime_name
        raw["_gmm_probs"] = probs.tolist()
        raw["_gmm_n_extreme_features"] = n_extreme

        logger.info(
            f"[GMM] {asset}: Regime={regime_name} conf={confidence:.3f} | "
            f"probs=[{', '.join(f'{p:.2f}' for p in probs)}] | "
            f"extreme_z={n_extreme}/{len(scaled)}"
        )
        if feature_names and len(feature_names) == len(features):
            feature_dump = {feature_names[i]: f"{features[i]:.4f}" for i in range(len(features))}
            scaled_dump = {feature_names[i]: f"{scaled[i]:.4f}" for i in range(len(features))}
            logger.debug(f"[GMM_DIAG] {asset}: Raw features: {feature_dump}")
            logger.debug(f"[GMM_DIAG] {asset}: Scaled features: {scaled_dump}")

        # Store GMM cache
        _gmm_raw_keys = {
            "_gmm_regime_idx": raw.get("_gmm_regime_idx"),
            "_gmm_regime_name": raw.get("_gmm_regime_name"),
            "_gmm_probs": raw.get("_gmm_probs"),
            "_gmm_n_extreme_features": raw.get("_gmm_n_extreme_features"),
        }
        self._ta_cache[_gmm_cache_key] = {
            "ts": _last_bar_ts, "n": n,
            "result": (confidence, regime_name),
            "raw_keys": _gmm_raw_keys,
        }

        return confidence, regime_name

    def _compute_strategy_signals(
        self,
        rsi_sig: float, macd_sig: float, bb_sig: float, ema_sig: float,
        adx_mult: float, rsi_val: float, adx_val: float, bb_pos: float,
        volume_ratio: float, regime: str,
    ) -> Dict[str, Dict]:
        """Compute independent alpha estimate per strategy (Best-of-N)."""
        strategies = {}

        # L4-13: Log unknown regime labels (once per unique label)
        if regime and regime not in _KNOWN_REGIMES and regime not in _LOGGED_UNKNOWN_REGIMES:
            logger.warning(f"[REGIME_UNKNOWN] Got '{regime}', using {_UNKNOWN_REGIME_FIT}x strength")
            _LOGGED_UNKNOWN_REGIMES.add(regime)

        # 1. Mean Reversion
        mr_sig = 0.0
        if rsi_val < 30:
            mr_sig = (30 - rsi_val) / 30
        elif rsi_val > 70:
            mr_sig = -(rsi_val - 70) / 30
        if abs(bb_pos) > 0.8:
            # [FIX-MR-RSI-CONFIRM] Require RSI confirmation for BB mean-revert.
            # Without this, BB fires LONG when price is below mean even if RSI is
            # neutral (30-70), producing mean-revert entries in downtrends.
            # SOL/BTC/ETH all lost -238/-72/-67 bps on BB-only mean_revert signals
            # in WEAK_CONSOLIDATION (2026-03-26). RSI was ~45 (neutral), MACD/EMA
            # bearish, but BB=+0.46 dominated.
            # When RSI extreme (<30 or >70): full BB power (1.0x)
            # When RSI neutral (30-70): reduce BB power to 0.4x
            _rsi_confirm = 1.0 if (rsi_val < 30 or rsi_val > 70) else 0.4
            mr_sig += -bb_pos * 0.3 * _rsi_confirm
        mr_sig = max(-1.0, min(1.0, mr_sig))
        _mr_table = REGIME_STRATEGY_FIT["mean_revert"]
        mr_fit = _mr_table.get(regime, _mr_table.get("_default", _UNKNOWN_REGIME_FIT))
        strategies["mean_revert"] = {"signal": mr_sig, "strength": abs(mr_sig) * mr_fit}

        # 2. Momentum
        # [FIX-EMA-CONSOLIDATION] In consolidation regimes, EMA_200 is a lagging
        # trap: after a large drop, price stabilizes but EMA stays high for weeks,
        # producing persistent false SHORT signals. Reduce EMA weight from 60%→30%
        # and increase MACD weight (a leading indicator) from 40%→70%.
        _CONSOLIDATION_REGIMES = {"QUIET_ACCUMULATION", "WEAK_CONSOLIDATION", "NEUTRAL_DRIFT"}
        if regime in _CONSOLIDATION_REGIMES:
            mom_sig = ema_sig * 0.3 + macd_sig * 0.7  # MACD leads, EMA lags
        else:
            mom_sig = ema_sig * 0.6 + macd_sig * 0.4  # Trending: EMA is valid
        if adx_val > 25:
            mom_sig *= 1.0 + (adx_val - 25) / 50
        else:
            mom_sig *= 0.5
        mom_sig = max(-1.0, min(1.0, mom_sig))
        _mom_table = REGIME_STRATEGY_FIT["momentum"]
        mom_fit = _mom_table.get(regime, _mom_table.get("_default", _UNKNOWN_REGIME_FIT))
        strategies["momentum"] = {"signal": mom_sig, "strength": abs(mom_sig) * mom_fit}

        # 3. Volume Breakout
        vb_sig = macd_sig
        if volume_ratio > 1.5:
            vb_sig *= 1.3
        elif volume_ratio < 0.5:
            vb_sig *= 0.5
        vb_sig = max(-1.0, min(1.0, vb_sig))
        _vb_table = REGIME_STRATEGY_FIT["volume_breakout"]
        vb_fit = _vb_table.get(regime, _vb_table.get("_default", _UNKNOWN_REGIME_FIT))
        strategies["volume_breakout"] = {"signal": vb_sig, "strength": abs(vb_sig) * vb_fit}

        # 4. VRP
        vrp_sig = 0.0
        if rsi_val > 60 and volume_ratio > 1.5 and bb_pos > 0.7:
            vrp_sig = -0.5 * (rsi_val - 60) / 40
        elif rsi_val < 40 and volume_ratio > 1.5 and bb_pos < -0.7:
            vrp_sig = 0.5 * (40 - rsi_val) / 40
        vrp_sig = max(-1.0, min(1.0, vrp_sig))
        _vrp_table = REGIME_STRATEGY_FIT["vrp"]
        vrp_fit = _vrp_table.get(regime, _vrp_table.get("_default", _UNKNOWN_REGIME_FIT))
        strategies["vrp"] = {"signal": vrp_sig, "strength": abs(vrp_sig) * vrp_fit}

        # 5. HOLD — no-trade strategy for directionless/choppy markets
        # Scores high when: low ADX (no trend), RSI neutral, volume thin.
        # Signal is always 0.0 (no direction). Strength comes from regime fitness.
        hold_base = 0.0
        if adx_val < 15:                                     # market has no trend
            hold_base += 0.25
        if 35 <= rsi_val <= 65 and abs(bb_pos) < 0.3:       # price near mean, no extreme
            hold_base += 0.20
        if volume_ratio < 0.7:                               # volume drying up
            hold_base += 0.10
        _hold_table = REGIME_STRATEGY_FIT["hold"]
        hold_fit = _hold_table.get(regime, _hold_table.get("_default", _UNKNOWN_REGIME_FIT))
        strategies["hold"] = {"signal": 0.0, "strength": hold_base * hold_fit}

        return strategies

    def generate_verification_data(self, asset: str) -> Dict[str, Any]:
        """Generate synthetic test data (all fields for constitution + indicators)."""
        base_prices = {"SOL": 185.0, "BTC": 95000.0, "ETH": 3500.0}
        base_price = base_prices.get(asset, 100.0)

        return {
            "asset": asset, "current_price": base_price,
            "prices": [base_price + i * 0.3 for i in range(100)],
            "volumes": [1000000.0 * (1 + 0.1 * (i % 10)) for i in range(100)],
            "volume_24h": 50000000.0, "data_age_seconds": 1.0, "data_age_sec": 1.0,
            "regime_state": "MOMENTUM_RALLY", "regime_direction": 0.72,
            "regime_confidence": 0.68,
            "crack_active": False, "crack_weight": 0.0,
            "quant_direction": 0.75, "quant_confidence": 0.70,
            "strategy_agreement": 0.5, "primary_strategy": "momentum",
            "lead_lag_edge": 35.0, "lead_lag_confidence": 0.72,
            "sol_signal_strength": 0.78, "signal_edge_bps": 55.0,
            "spread_bps": 8.0, "orderbook_depth": 1.2,
            "estimated_friction_bps": 12.0,
            "current_exposure": 0.0, "position_direction": 0,
            "current_tranche": 0, "current_drawdown": 0.02,
            "cross_asset_correlation": 0.0,
            "sol_exposure": 0.0, "btc_exposure": 0.0, "eth_exposure": 0.0,
            "no_trade_triggers": {},
            "is_weekend": self._weekend_manager.update().is_weekend if self._weekend_manager else False,
            "is_holiday": False, "is_low_liquidity": False,
            "data_valid": True,
            "market_data_fetched_at_ts": _time.time(),
            "dvol_zscore": 0.5, "vpin": 0.35, "vpin_source": "synthetic", "ofi_zscore": 0.0,
            "correlation_btc_eth_sol": 0.72,
            "orderbook_depth_1pct_usd": 500000.0,
            "orderbook_stale": False,
            "orderbook_fallback_reason": "live",
            "orderbook_cache_age_seconds": 0.0,
            "close": base_price,
            "rsi": 62.0, "macd": base_price * 0.002,
            "macd_signal": base_price * 0.0015,
            "macd_hist": base_price * 0.0005,
            "bb_upper": base_price * 1.04, "bb_middle": base_price * 1.0,
            "bb_lower": base_price * 0.96, "bb_width": 0.08,
            "ema_200": base_price * 0.97, "sma_50": base_price * 0.99,
            "atr": base_price * 0.02, "adx": 28.0,
            "di_plus": 24.0, "di_minus": 18.0,
            "volume_sma": 900000.0, "volume_ratio": 1.1, "atr_ratio": 1.05,
            "rsi_14": 0.12, "macd_12_26": 0.002,
            "bb_width_20": 0.08, "atr_14": 0.02, "vol_ratio_s": 1.1,
            "rsi_14_denoised": 0.10, "macd_12_26_denoised": 0.0018,
            "bb_width_20_denoised": 0.075, "atr_14_denoised": 0.018,
            "vol_ratio_s_denoised": 1.05,
            "price_percentile_20d": 0.65,
            "funding_rate": 0.0001, "funding_bias": 0.33,
            "funding_rate_zscore": 0.35, "oi_change_5d": 0.12,
            "liq_imbalance": 0.15, "taker_ratio_zscore": 0.20,
            "tradecount_zscore": 0.18, "taker_vol_momentum": 0.10,
            "has_external_data": 1.0,
            "recent_trade_count": 320.0, "recent_taker_volume_usd": 1_500_000.0,
            "trade_metrics_source": "synthetic",
            "sentiment_zscore": 0.9, "fear_greed_index": 0.65,
        }

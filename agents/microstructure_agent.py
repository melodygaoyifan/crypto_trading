"""
================================================================================
HMATS v6 - Microstructure Agent (TRIGGER / ADVISE / EXEC)
================================================================================
Per-asset cross-exchange microstructure signals.  Does NOT produce DECIDE-level
direction - only TRIGGER/ADVISE evidence consumed by Authority Fusion and the
execution layer.

Features:
  1. Cross-exchange price lag detection (lead_lag_edge, lead_lag_confidence)
  2. Order flow imbalance (order_book_imbalance, cvd_divergence proxy)
  3. Spread dynamics (spread_bps)
  4. Taker flow spike detection

Public interface:
  - MicrostructureArbitrageAgent.generate_signal(asset, market_data, regime)
        -> Dict[str, Any]   (v6-compatible, never None, never raises)
  - get_microstructure_agent() / reset_microstructure_agent()  (singleton)
  - MicrostructureEventBusIntegration  (optional EventBus wiring)

Authority: TRIGGER / ADVISE / EXEC  (does NOT overwrite Quant DECIDE direction)
================================================================================
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from datetime import datetime, timezone
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class MicroSignalType(Enum):
    """Types of microstructure signals."""
    LAG_FOLLOW = "lag_follow"
    SPREAD_COMPRESS = "spread_compress"
    SPREAD_EXPAND = "spread_expand"
    TAKER_SPIKE = "taker_spike"
    IMBALANCE_BUY = "imbalance_buy"
    IMBALANCE_SELL = "imbalance_sell"
    LATENCY_ARB = "latency_arb"
    FUNDING_PREMIUM = "funding_premium"


class ExchangeRole(Enum):
    """Role of exchange in price discovery."""
    LEADER = "leader"
    FOLLOWER = "follower"
    NEUTRAL = "neutral"


@dataclass
class ExchangeSnapshot:
    """Point-in-time exchange state."""
    exchange: str
    timestamp_ms: float
    bid: float
    ask: float
    mid: float
    spread_bps: float
    bid_size: float
    ask_size: float
    last_trade_price: float
    last_trade_size: float
    last_trade_side: str  # "buy" or "sell"

    @property
    def imbalance(self) -> float:
        """Order book imbalance: positive = more bids, negative = more asks."""
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total


@dataclass
class CrossExchangeState:
    """State across multiple exchanges."""
    timestamp: datetime
    snapshots: Dict[str, ExchangeSnapshot]

    best_bid_exchange: str = ""
    best_ask_exchange: str = ""
    cross_spread_bps: float = 0.0
    price_leader: str = ""

    def compute_metrics(self):
        """Compute cross-exchange metrics."""
        if len(self.snapshots) < 2:
            return

        best_bid = -float('inf')
        best_ask = float('inf')

        for name, snap in self.snapshots.items():
            if snap.bid > best_bid:
                best_bid = snap.bid
                self.best_bid_exchange = name
            if snap.ask < best_ask:
                best_ask = snap.ask
                self.best_ask_exchange = name

        mid = (best_bid + best_ask) / 2
        if mid > 0:
            self.cross_spread_bps = (best_ask - best_bid) / mid * 10000


@dataclass
class MicrostructureSignal:
    """Internal trading signal from microstructure analysis."""
    timestamp: datetime
    signal_type: MicroSignalType
    direction: float  # -1 to 1
    confidence: float  # 0 to 1
    urgency: float  # 0 to 1
    asset: str
    reasoning: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure agent."""
    leader_exchange: str = "binance"
    follower_exchange: str = "kraken"

    # Lag detection
    lag_threshold_ms: float = 100.0
    min_price_diff_bps: float = 5.0

    # Taker flow
    taker_spike_sigma: float = 3.0
    taker_lookback: int = 100

    # Imbalance
    imbalance_threshold: float = 0.3
    imbalance_lookback: int = 20

    # Spread
    spread_compression_pct: float = 0.3
    spread_expansion_pct: float = 0.5

    # Signal generation
    min_confidence: float = 0.5
    signal_cooldown_ms: float = 500.0

    # History
    max_history: int = 1000

    # Safety
    min_samples: int = 5
    max_snapshot_age_ms: float = 5000.0  # 5s freshness


# =============================================================================
# LAG DETECTOR
# =============================================================================

class LagDetector:
    """Detects price leads/lags between exchanges."""

    PRICE_HISTORY_MAXLEN = 500  # [P371] one constant; the restore path builds the same deque

    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.price_history: Dict[str, Deque] = {}
        self.lag_estimates: Dict[str, float] = {}

    def update(self, exchange: str, timestamp_ms: float, price: float):
        if exchange not in self.price_history:
            self.price_history[exchange] = deque(maxlen=self.PRICE_HISTORY_MAXLEN)  # [P371]
        self.price_history[exchange].append((timestamp_ms, price))

    def estimate_lag(self, leader: str, follower: str) -> Optional[float]:
        if leader not in self.price_history or follower not in self.price_history:
            return None

        leader_data = list(self.price_history[leader])
        follower_data = list(self.price_history[follower])

        if len(leader_data) < 50 or len(follower_data) < 50:
            return None

        leader_times = np.array([d[0] for d in leader_data])
        leader_prices = np.array([d[1] for d in leader_data])
        follower_times = np.array([d[0] for d in follower_data])
        follower_prices = np.array([d[1] for d in follower_data])

        best_lag = 0.0
        best_corr = -1.0

        for lag_ms in range(-200, 201, 10):
            shifted = follower_times + lag_ms
            try:
                interp_leader = np.interp(shifted, leader_times, leader_prices)
                if len(interp_leader) > 0:
                    corr = np.corrcoef(follower_prices, interp_leader)[0, 1]
                    if not np.isnan(corr) and corr > best_corr:
                        best_corr = corr
                        best_lag = lag_ms
            except Exception:
                continue

        if best_corr > 0.8:
            self.lag_estimates[follower] = best_lag
            return best_lag

        return None

    def detect_lag_opportunity(self, leader: str, follower: str,
                               current_diff_bps: float) -> Optional[Tuple[float, float]]:
        lag_ms = self.lag_estimates.get(follower)

        if lag_ms is None or abs(lag_ms) < self.config.lag_threshold_ms:
            return None

        if abs(current_diff_bps) < self.config.min_price_diff_bps:
            return None

        direction = 1.0 if current_diff_bps > 0 else -1.0
        confidence = min(1.0, abs(current_diff_bps) / 20.0) * 0.7
        confidence += min(0.3, abs(lag_ms) / 200.0 * 0.3)

        return direction, confidence


# =============================================================================
# TAKER FLOW ANALYZER
# =============================================================================

class TakerFlowAnalyzer:
    """Analyzes taker order flow for signals."""

    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.taker_buys: Deque = deque(maxlen=config.taker_lookback)
        self.taker_sells: Deque = deque(maxlen=config.taker_lookback)
        self.buy_volume_history: Deque = deque(maxlen=config.taker_lookback)
        self.sell_volume_history: Deque = deque(maxlen=config.taker_lookback)

    def add_trade(self, side: str, size: float, timestamp_ms: float):
        if side == "buy":
            self.taker_buys.append((timestamp_ms, size))
            self.buy_volume_history.append(size)
        else:
            self.taker_sells.append((timestamp_ms, size))
            self.sell_volume_history.append(size)

    def detect_spike(self) -> Optional[Tuple[str, float]]:
        if len(self.buy_volume_history) < 20 or len(self.sell_volume_history) < 20:
            return None

        buy_arr = np.array(list(self.buy_volume_history))
        sell_arr = np.array(list(self.sell_volume_history))

        buy_mean = np.mean(buy_arr[:-1]) if len(buy_arr) > 1 else 0
        buy_std = np.std(buy_arr[:-1]) if len(buy_arr) > 1 else 1
        sell_mean = np.mean(sell_arr[:-1]) if len(sell_arr) > 1 else 0
        sell_std = np.std(sell_arr[:-1]) if len(sell_arr) > 1 else 1

        buy_z = (buy_arr[-1] - buy_mean) / buy_std if buy_std > 0 else 0
        sell_z = (sell_arr[-1] - sell_mean) / sell_std if sell_std > 0 else 0

        threshold = self.config.taker_spike_sigma

        if buy_z > threshold and buy_z > sell_z:
            return "buy", buy_z
        elif sell_z > threshold and sell_z > buy_z:
            return "sell", sell_z

        return None

    def get_imbalance(self) -> float:
        recent_buy = sum(v for _, v in list(self.taker_buys)[-10:])
        recent_sell = sum(v for _, v in list(self.taker_sells)[-10:])
        total = recent_buy + recent_sell
        if total == 0:
            return 0.0
        return (recent_buy - recent_sell) / total


# =============================================================================
# PER-ASSET STATE CONTAINER
# =============================================================================

class _AssetState:
    """All mutable state for one asset.  Isolates BTC/ETH/SOL completely."""

    def __init__(self, config: MicrostructureConfig):
        self.lag_detector = LagDetector(config)
        self.taker_analyzer = TakerFlowAnalyzer(config)
        self.exchange_states: Dict[str, ExchangeSnapshot] = {}
        self.cross_state: Optional[CrossExchangeState] = None
        self.spread_history: Deque = deque(maxlen=100)
        self.imbalance_history: Deque = deque(maxlen=config.imbalance_lookback)
        self.last_signal_time: float = 0.0
        self.signals_generated: Deque = deque(maxlen=100)
        self.last_v6_payload: Dict[str, Any] = {}


# =============================================================================
# NEUTRAL PAYLOAD HELPER
# =============================================================================

def _neutral_v6_payload(asset: str) -> Dict[str, Any]:
    """Return a fail-safe neutral v6 payload - no signal, no trigger."""
    now = time.time()
    return {
        # Canonical v6 keys consumed by integration_v36 / Authority Fusion
        "lead_lag_edge": 0.0,
        "lead_lag_confidence": 0.0,
        "cvd_divergence": 0.0,
        "order_book_imbalance": 0.0,
        "spread_bps": 0.0,
        # Composite direction (ADVISE, not DECIDE)
        "micro_direction": 0.0,
        "micro_confidence": 0.0,
        "micro_urgency": 0.0,
        "micro_primary_signal": "none",
        # Quality / metadata
        "micro_data_quality": 0.0,
        "micro_is_valid": True,
        "data_age_seconds": 0.0,
        "asof_timestamp": now,
        "asset": asset,
        "diagnostics": {},
    }


# =============================================================================
# [UPG5] CROSS-ASSET LAG DETECTOR
# =============================================================================

class CrossAssetLagDetector:
    """
    [UPG5] BTC->ETH/SOL lead-lag detector.

    BTC drops -> ETH/SOL follow 1-4H later.
    If BTC fell but follower hasn't -> short follower signal.

    ADVISE-level only - feeds through Authority Fusion.
    """

    def __init__(self, lookback_bars: int = 6, btc_threshold: float = -0.02):
        self._returns: Dict[str, deque] = {
            "BTC": deque(maxlen=lookback_bars),
            "ETH": deque(maxlen=lookback_bars),
            "SOL": deque(maxlen=lookback_bars),
        }
        self._prev_prices: Dict[str, float] = {}
        self._btc_threshold = btc_threshold  # BTC drop threshold (default -2%)
        self._lag_max_bars = 2  # Max 2 bars (8h) follow-through

    def update_price(self, asset: str, price: float):
        """Call once per tick per asset. Records return."""
        if asset not in self._returns:
            return
        prev = self._prev_prices.get(asset)
        if prev and prev > 0:
            ret = (price - prev) / prev
            self._returns[asset].append(ret)
        self._prev_prices[asset] = price

    def detect_lag_signal(self, target_asset: str) -> Optional[Dict]:
        """
        Detect BTC->follower lag opportunity.
        Returns signal dict or None.
        """
        if target_asset == "BTC":
            return None

        btc_rets = list(self._returns.get("BTC", []))
        target_rets = list(self._returns.get(target_asset, []))

        if len(btc_rets) < 2 or len(target_rets) < 2:
            return None

        # BTC cumulative return over last 2 bars
        btc_cum_2bar = sum(btc_rets[-2:])
        target_cum_2bar = sum(target_rets[-2:])

        # Condition: BTC down significantly, target hasn't followed
        if btc_cum_2bar < self._btc_threshold and target_cum_2bar > self._btc_threshold * 0.3:
            lag_gap = abs(btc_cum_2bar - target_cum_2bar)
            confidence = min(0.75, lag_gap * 10)  # 10% gap -> max confidence

            return {
                "asset": target_asset,
                "direction": -1.0,
                "confidence": round(confidence, 3),
                "signal_type": "CROSS_ASSET_LAG",
                "reasoning": (
                    f"BTC {btc_cum_2bar:+.1%} (2-bar) but {target_asset} "
                    f"{target_cum_2bar:+.1%}, expect follow-through"
                ),
                "meta": {
                    "btc_cum_return": round(btc_cum_2bar, 4),
                    "target_cum_return": round(target_cum_2bar, 4),
                    "lag_gap": round(lag_gap, 4),
                },
            }

        return None


# =============================================================================
# MICROSTRUCTURE AGENT (V6)
# =============================================================================

class MicrostructureArbitrageAgent:
    """
    HMATS v6 microstructure agent - TRIGGER / ADVISE / EXEC authority.

    Per-asset state isolation.  No background loops.  generate_signal() returns
    a flat v6-compatible dict (never None, never raises).
    """

    ASSETS = ("BTC", "ETH", "SOL")

    # [P371] ---- warmup-sample persistence -----------------------------------
    # [P371] The `min_samples` gate counts spread_history + imbalance_history +
    # [P371] the lag detector's per-exchange price_history. All three were
    # [P371] per-process deques, so every restart put the agent back at
    # [P371] `insufficient_samples` (dq 0.7-0.8, measured P316/P370) — the
    # [P371] P301/P316 class, one more location. Deliberately reuses
    # [P371] strategies/_warmup_state (one atomic writer, one staleness rule,
    # [P371] one set of fail directions — P172). Decision logic and thresholds
    # [P371] are UNTOUCHED: only where the samples live across restarts.
    _WARMUP_STATE_NAME = "micro_agent_samples"          # [P371]
    _WARMUP_MAX_AGE_SEC = 7 * 24 * 3600.0               # [P371] same bound as the pipeline buffers (P316)

    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()
        self._per_asset: Dict[str, _AssetState] = {}
        self._tick_count = 0
        self._cross_asset_lag = CrossAssetLagDetector()  # [UPG5]
        self._restore_warmup_samples()  # [P371] before the first tick; fail-soft
        logger.info("MicrostructureArbitrageAgent v6 initialized")

    # ---- per-asset state accessor ------------------------------------------

    def _state(self, asset: str) -> _AssetState:
        if asset not in self._per_asset:
            self._per_asset[asset] = _AssetState(self.config)
        return self._per_asset[asset]

    # ---- [P371] warmup-sample persistence ----------------------------------

    def _warmup_series(self) -> Dict[str, List[float]]:
        """[P371] Flatten the sample deques to {composite_key: [floats]}.

        [P371] The helper carries float lists only, so the lag detector's
        [P371] (timestamp_ms, price) tuples travel as two parallel series
        [P371] (`lag_ts::A::EX` / `lag_px::A::EX`) and are zipped back on
        [P371] restore; a length mismatch drops that exchange rather than
        [P371] pairing the wrong timestamp with a price.
        """
        series: Dict[str, List[float]] = {}  # [P371]
        for asset, st in self._per_asset.items():  # [P371]
            if len(st.spread_history):  # [P371]
                series[f"spread::{asset}"] = [float(v) for v in st.spread_history]  # [P371]
            if len(st.imbalance_history):  # [P371]
                series[f"imbalance::{asset}"] = [float(v) for v in st.imbalance_history]  # [P371]
            for ex, dq in st.lag_detector.price_history.items():  # [P371]
                if len(dq):  # [P371]
                    series[f"lag_ts::{asset}::{ex}"] = [float(t) for t, _ in dq]  # [P371]
                    series[f"lag_px::{asset}::{ex}"] = [float(p) for _, p in dq]  # [P371]
        return series  # [P371]

    def _restore_warmup_samples(self) -> None:
        """[P371] Restore the sample deques saved by the previous process.

        [P371] Every failure path restores NOTHING and the agent warms up
        [P371] exactly as it did before P371: a missing/corrupt/version-
        [P371] mismatched/stale file is a cold start, logged rather than a
        [P371] silent return (a cold start that announces itself is the
        [P371] difference between a warmup and a permanent NEUTRAL, P316).
        """
        try:  # [P371]
            from strategies._warmup_state import load as _wload  # [P371]
            saved = _wload(self._WARMUP_STATE_NAME,  # [P371]
                           max_age_sec=self._WARMUP_MAX_AGE_SEC)  # [P371]
            if not saved:  # [P371]
                logger.info("[MicroV6] warmup samples: no saved state — cold "  # [P371]
                            "start, warming up from scratch")  # [P371]
                return  # [P371]
            restored: Dict[str, int] = {}  # [P371]
            lag_ts: Dict[tuple, list] = {}  # [P371]
            lag_px: Dict[tuple, list] = {}  # [P371]
            for key, vals in saved.items():  # [P371]
                parts = str(key).split("::")  # [P371]
                kind = parts[0]  # [P371]
                if kind in ("spread", "imbalance") and len(parts) == 2:  # [P371]
                    st = self._state(parts[1])  # [P371]
                    dq = (st.spread_history if kind == "spread"  # [P371]
                          else st.imbalance_history)  # [P371]
                    tail = list(vals)[-dq.maxlen:] if dq.maxlen else list(vals)  # [P371]
                    for v in tail:  # [P371]
                        dq.append(float(v))  # [P371]
                    restored[parts[1]] = restored.get(parts[1], 0) + len(tail)  # [P371]
                elif kind == "lag_ts" and len(parts) == 3:  # [P371]
                    lag_ts[(parts[1], parts[2])] = list(vals)  # [P371]
                elif kind == "lag_px" and len(parts) == 3:  # [P371]
                    lag_px[(parts[1], parts[2])] = list(vals)  # [P371]
                # [P371] unknown keys are ignored: an old encoding must not
                # [P371] land in a deque it was never meant for
            for (asset, ex), ts in lag_ts.items():  # [P371]
                px = lag_px.get((asset, ex))  # [P371]
                if px is None or len(px) != len(ts):  # [P371]
                    logger.warning("[MicroV6] warmup samples: lag series %s/%s "  # [P371]
                                   "ts/px length mismatch — dropped, not paired",  # [P371]
                                   asset, ex)  # [P371]
                    continue  # [P371]
                st = self._state(asset)  # [P371]
                dq = st.lag_detector.price_history.setdefault(  # [P371]
                    ex, deque(maxlen=LagDetector.PRICE_HISTORY_MAXLEN))  # [P371]
                # [P370] slice by the int constant, not dq.maxlen: deque.maxlen
                # is typed `int | None`, so `-dq.maxlen` is a mypy [operator]
                # finding even though this deque is always bounded (P284b:
                # fix at source, never baseline). Same bound, typed honestly.
                pairs = list(zip(ts, px))[-LagDetector.PRICE_HISTORY_MAXLEN:]  # [P371]
                for t, p in pairs:  # [P371]
                    dq.append((float(t), float(p)))  # [P371]
                restored[asset] = restored.get(asset, 0) + len(pairs)  # [P371]
            if restored:  # [P371]
                logger.info("[MicroV6] warmup samples: restored %s",  # [P371]
                            ", ".join(f"{a}={n}" for a, n in sorted(restored.items())))  # [P371]
            else:  # [P371]
                logger.info("[MicroV6] warmup samples: saved state held no usable "  # [P371]
                            "series — cold start, warming up from scratch")  # [P371]
        except Exception as e:  # noqa: silent-swallow — logged; cold start is pre-P371 behaviour  # [P371]
            logger.warning("[MicroV6] warmup samples: restore failed (%s: %s) — "  # [P371]
                           "cold start, warming up from scratch",  # [P371]
                           type(e).__name__, e)  # [P371]

    def _persist_warmup_samples(self) -> None:
        """[P371] Save the sample deques. Never raises: a warmup that cannot
        [P371] be saved must not take a tick down; the only cost of a failed
        [P371] save is the warmup this exists to end.
        """
        try:  # [P371]
            from strategies._warmup_state import save as _wsave  # [P371]
            series = self._warmup_series()  # [P371]
            if series:  # [P371]
                _wsave(self._WARMUP_STATE_NAME, series)  # [P371]
        except Exception as e:  # noqa: silent-swallow — logged; persistence only  # [P371]
            logger.debug("[MicroV6] warmup samples: persist skipped (%s: %s)",  # [P371]
                         type(e).__name__, e)  # [P371]

    # ---- data ingestion (asset-scoped) -------------------------------------

    def update_exchange(
        self,
        asset: str,
        exchange: str,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        last_price: float = None,
        last_size: float = 0,
        last_side: str = "",
        timestamp_ms: float = None,
    ):
        """
        Update exchange state for a specific asset.

        Args:
            asset: "BTC", "ETH", or "SOL"
            exchange: "binance", "kraken", etc.
            bid/ask/bid_size/ask_size: top-of-book snapshot
            last_price/last_size/last_side: most recent trade (optional)
            timestamp_ms: exchange timestamp in ms (defaults to now)
        """
        st = self._state(asset)
        now_ms = timestamp_ms or (time.time() * 1000)
        mid = (bid + ask) / 2
        spread_bps = (ask - bid) / mid * 10000 if mid > 0 else 0

        snapshot = ExchangeSnapshot(
            exchange=exchange,
            timestamp_ms=now_ms,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_bps=spread_bps,
            bid_size=bid_size,
            ask_size=ask_size,
            last_trade_price=last_price or mid,
            last_trade_size=last_size,
            last_trade_side=last_side,
        )

        st.exchange_states[exchange] = snapshot
        st.lag_detector.update(exchange, now_ms, mid)

        if last_side and last_size > 0:
            st.taker_analyzer.add_trade(last_side, last_size, now_ms)

        self._update_cross_state(st)
        self._persist_warmup_samples()  # [P371] after each append; fail-soft

    # ---- cross-exchange state update ---------------------------------------

    @staticmethod
    def _update_cross_state(st: _AssetState):
        if len(st.exchange_states) < 2:
            return

        st.cross_state = CrossExchangeState(
            timestamp=datetime.now(timezone.utc),
            snapshots=st.exchange_states.copy(),
        )
        st.cross_state.compute_metrics()

        if st.cross_state.cross_spread_bps != 0:
            st.spread_history.append(st.cross_state.cross_spread_bps)

        total_imb = sum(s.imbalance for s in st.exchange_states.values())
        st.imbalance_history.append(total_imb / len(st.exchange_states))

    # ---- v6 public interface -----------------------------------------------

    def generate_signal(
        self,
        asset: str = "BTC",
        market_data: Optional[Dict] = None,
        regime: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate v6-compatible microstructure signal for *asset*.

        Never returns None.  Never raises.  Neutral payload on any error or
        insufficient data.

        Canonical output keys (consumed by Authority Fusion / execution):
          lead_lag_edge (bps), lead_lag_confidence, cvd_divergence,
          order_book_imbalance, spread_bps, micro_direction, micro_confidence,
          micro_urgency, micro_primary_signal, micro_data_quality,
          micro_is_valid, data_age_seconds, asof_timestamp, asset, diagnostics.
        """
        try:
            self._tick_count += 1
            now = time.time()
            now_ms = now * 1000
            st = self._state(asset)

            # --- If market_data provided, ingest it as exchange snapshot -----
            if market_data:
                self._ingest_market_data(asset, market_data, now_ms)

            # --- Freshness guard --------------------------------------------
            if not st.exchange_states:
                p = _neutral_v6_payload(asset)
                p["diagnostics"] = {"reason": "no_exchange_data"}
                st.last_v6_payload = p
                return p

            newest_ms = max(s.timestamp_ms for s in st.exchange_states.values())
            age_ms = now_ms - newest_ms
            if age_ms > self.config.max_snapshot_age_ms:
                p = _neutral_v6_payload(asset)
                p["data_age_seconds"] = age_ms / 1000
                p["diagnostics"] = {"reason": "stale_snapshot",
                                    "age_ms": round(age_ms, 1)}
                st.last_v6_payload = p
                return p

            # --- Minimum samples guard --------------------------------------
            samples = (len(st.spread_history) + len(st.imbalance_history)
                       + sum(len(d.price_history.get(e, []))
                             for d in [st.lag_detector]
                             for e in st.exchange_states))
            if samples < self.config.min_samples:
                p = _neutral_v6_payload(asset)
                p["micro_data_quality"] = min(samples / max(self.config.min_samples, 1), 1.0)
                p["diagnostics"] = {"reason": "insufficient_samples",
                                    "samples": samples}
                st.last_v6_payload = p
                return p

            # --- Run internal signal checks ---------------------------------
            internal_sig = self._generate_internal(asset, st, now_ms)

            # --- Build v6 payload -------------------------------------------
            p = self._build_v6_payload(asset, st, internal_sig, now, age_ms)
            st.last_v6_payload = p
            return p

        except Exception as exc:
            logger.error("[MicroV6] generate_signal(%s) failed: %s",
                         asset, exc, exc_info=True)
            p = _neutral_v6_payload(asset)
            p["diagnostics"] = {"error": str(exc)}
            return p

    # ---- market_data ingestion helper --------------------------------------

    def _ingest_market_data(self, asset: str, md: Dict, now_ms: float):
        """
        If caller passes a market_data dict (e.g. from the 4H tick), extract
        any LOB / trade fields and feed them into the per-asset state.
        """
        # Leader (Binance) snapshot from market_data
        leader = self.config.leader_exchange
        follower = self.config.follower_exchange

        # Try follower exchange snapshot from market_data
        bid = md.get("bid") or md.get(f"{asset.lower()}_bid", 0)
        ask = md.get("ask") or md.get(f"{asset.lower()}_ask", 0)
        if bid > 0 and ask > 0:
            bid_sz = md.get("bid_size", md.get("bid_depth", 1.0))
            ask_sz = md.get("ask_size", md.get("ask_depth", 1.0))
            self.update_exchange(
                asset, follower,
                bid=bid, ask=ask,
                bid_size=bid_sz, ask_size=ask_sz,
                timestamp_ms=md.get("timestamp_ms", now_ms),
            )

        # Leader snapshot (if available)
        leader_bid = md.get(f"{leader}_bid", 0)
        leader_ask = md.get(f"{leader}_ask", 0)
        if leader_bid > 0 and leader_ask > 0:
            self.update_exchange(
                asset, leader,
                bid=leader_bid, ask=leader_ask,
                bid_size=md.get(f"{leader}_bid_size", 1.0),
                ask_size=md.get(f"{leader}_ask_size", 1.0),
                timestamp_ms=md.get("timestamp_ms", now_ms),
            )

        # [FIX-AG12] Ingest taker trade data from market_data so CVD divergence works.
        # Without this, taker_buys/taker_sells deques stay empty -> cvd_divergence always 0.
        st = self._state(asset)
        _last_side = md.get("last_trade_side", md.get("taker_side", ""))
        _last_size = float(md.get("last_trade_size", md.get("taker_volume", 0)) or 0)
        if _last_side and _last_size > 0:
            st.taker_analyzer.add_trade(_last_side, _last_size, now_ms)
        # Also ingest aggregated taker volumes if available
        _taker_buy_vol = float(md.get("taker_buy_volume", 0) or 0)
        _taker_sell_vol = float(md.get("taker_sell_volume", 0) or 0)
        if _taker_buy_vol > 0:
            st.taker_analyzer.add_trade("buy", _taker_buy_vol, now_ms)
        if _taker_sell_vol > 0:
            st.taker_analyzer.add_trade("sell", _taker_sell_vol, now_ms)

    # ---- internal signal generation ----------------------------------------

    def _generate_internal(
        self, asset: str, st: _AssetState, now_ms: float,
    ) -> Optional[MicrostructureSignal]:
        """Run the four strategy checks, return first hit or None."""
        if now_ms - st.last_signal_time < self.config.signal_cooldown_ms:
            return None

        for checker in (self._check_lag_opportunity,
                        self._check_taker_spike,
                        self._check_imbalance,
                        self._check_spread_dynamics):
            sig = checker(asset, st)
            if sig is not None:
                st.last_signal_time = now_ms
                st.signals_generated.append(sig)
                return sig

        return None

    def _check_lag_opportunity(self, asset: str, st: _AssetState) -> Optional[MicrostructureSignal]:
        leader = self.config.leader_exchange
        follower = self.config.follower_exchange

        if leader not in st.exchange_states or follower not in st.exchange_states:
            return None

        leader_mid = st.exchange_states[leader].mid
        follower_mid = st.exchange_states[follower].mid
        if leader_mid == 0 or follower_mid == 0:
            return None

        diff_bps = (leader_mid - follower_mid) / leader_mid * 10000
        result = st.lag_detector.detect_lag_opportunity(leader, follower, diff_bps)
        if result is None:
            return None

        direction, confidence = result
        if confidence < self.config.min_confidence:
            return None

        return MicrostructureSignal(
            timestamp=datetime.now(timezone.utc),
            signal_type=MicroSignalType.LAG_FOLLOW,
            direction=direction,
            confidence=confidence,
            urgency=0.8,
            asset=asset,
            reasoning=f"Price lag: {leader} leads {follower} by {abs(diff_bps):.1f}bps",
            metadata={"leader": leader, "follower": follower,
                      "diff_bps": diff_bps,
                      "estimated_lag_ms": st.lag_detector.lag_estimates.get(follower, 0)},
        )

    def _check_taker_spike(self, asset: str, st: _AssetState) -> Optional[MicrostructureSignal]:
        result = st.taker_analyzer.detect_spike()
        if result is None:
            return None

        side, z_score = result
        direction = 1.0 if side == "buy" else -1.0
        confidence = min(1.0, z_score / 5.0)
        if confidence < self.config.min_confidence:
            return None

        return MicrostructureSignal(
            timestamp=datetime.now(timezone.utc),
            signal_type=MicroSignalType.TAKER_SPIKE,
            direction=direction,
            confidence=confidence,
            urgency=0.9,
            asset=asset,
            reasoning=f"Taker {side} spike: z={z_score:.2f}",
            metadata={"side": side, "z_score": z_score},
        )

    def _check_imbalance(self, asset: str, st: _AssetState) -> Optional[MicrostructureSignal]:
        if len(st.imbalance_history) < 5:
            return None

        recent = list(st.imbalance_history)[-5:]
        avg_imb = float(np.mean(recent))

        if abs(avg_imb) < self.config.imbalance_threshold:
            return None

        if avg_imb > 0:
            if not all(i > 0 for i in recent):
                return None
            sig_type = MicroSignalType.IMBALANCE_BUY
            direction = 0.6
        else:
            if not all(i < 0 for i in recent):
                return None
            sig_type = MicroSignalType.IMBALANCE_SELL
            direction = -0.6

        confidence = min(1.0, abs(avg_imb) / 0.5) * 0.7
        if confidence < self.config.min_confidence:
            return None

        return MicrostructureSignal(
            timestamp=datetime.now(timezone.utc),
            signal_type=sig_type,
            direction=direction,
            confidence=confidence,
            urgency=0.5,
            asset=asset,
            reasoning=f"Persistent OB imbalance: {avg_imb:.2f}",
            metadata={"avg_imbalance": avg_imb},
        )

    def _check_spread_dynamics(self, asset: str, st: _AssetState) -> Optional[MicrostructureSignal]:
        if len(st.spread_history) < 20:
            return None

        spreads = np.array(list(st.spread_history))
        recent = spreads[-5:]
        historical = spreads[:-5]
        avg_recent = float(np.mean(recent))
        avg_hist = float(np.mean(historical))

        if avg_hist == 0:
            return None

        change_pct = (avg_recent - avg_hist) / avg_hist

        if change_pct < -self.config.spread_compression_pct:
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type=MicroSignalType.SPREAD_COMPRESS,
                direction=0.0,
                confidence=0.6,
                urgency=0.4,
                asset=asset,
                reasoning=f"Spread compressed {abs(change_pct)*100:.0f}%",
                metadata={"change_pct": change_pct, "spread_bps": avg_recent},
            )
        elif change_pct > self.config.spread_expansion_pct:
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type=MicroSignalType.SPREAD_EXPAND,
                direction=0.0,
                confidence=0.5,
                urgency=0.6,
                asset=asset,
                reasoning=f"Spread expanded {change_pct*100:.0f}%",
                metadata={"change_pct": change_pct, "spread_bps": avg_recent},
            )

        return None

    # ---- v6 payload builder ------------------------------------------------

    def _build_v6_payload(
        self,
        asset: str,
        st: _AssetState,
        sig: Optional[MicrostructureSignal],
        now: float,
        age_ms: float,
    ) -> Dict[str, Any]:
        """Convert internal state + optional signal into canonical v6 dict."""

        # Lead-lag edge
        leader = self.config.leader_exchange
        follower = self.config.follower_exchange
        lead_lag_edge = 0.0
        lead_lag_conf = 0.0
        if leader in st.exchange_states and follower in st.exchange_states:
            l_mid = st.exchange_states[leader].mid
            f_mid = st.exchange_states[follower].mid
            if l_mid > 0 and f_mid > 0:
                lead_lag_edge = (l_mid - f_mid) / l_mid * 10000
                lag_ms = st.lag_detector.lag_estimates.get(follower)
                if lag_ms is not None and abs(lag_ms) >= self.config.lag_threshold_ms:
                    lead_lag_conf = min(1.0, abs(lead_lag_edge) / 20.0)

        # CVD divergence proxy from taker imbalance
        cvd_div = st.taker_analyzer.get_imbalance()

        # Order book imbalance (average across exchanges for this asset)
        ob_imb = 0.0
        if st.exchange_states:
            ob_imb = sum(s.imbalance for s in st.exchange_states.values()) / len(st.exchange_states)

        # Spread from follower (execution venue)
        spread = 0.0
        if follower in st.exchange_states:
            spread = st.exchange_states[follower].spread_bps

        # Data quality heuristic
        n_exchanges = len(st.exchange_states)
        has_leader = leader in st.exchange_states
        has_history = len(st.spread_history) >= 10
        quality = (0.3 * min(n_exchanges / 2, 1.0)
                   + 0.3 * float(has_leader)
                   + 0.2 * float(has_history)
                   + 0.2 * min(len(st.imbalance_history) / 10, 1.0))

        # Composite direction/confidence from internal signal
        direction = 0.0
        confidence = 0.0
        urgency = 0.0
        primary = "none"
        diag: Dict[str, Any] = {
            "tick": self._tick_count,
            "n_exchanges": n_exchanges,
            "has_leader": has_leader,
        }

        if sig is not None:
            direction = sig.direction
            confidence = sig.confidence
            urgency = sig.urgency
            primary = sig.signal_type.value
            diag["reasoning"] = sig.reasoning
            diag.update(sig.metadata)

        # [UPG5] Cross-asset lag detection
        _lag_signal = None
        try:
            # Update price for this asset (use follower mid as canonical price)
            if follower in st.exchange_states and st.exchange_states[follower].mid > 0:
                self._cross_asset_lag.update_price(asset, st.exchange_states[follower].mid)
            _lag_signal = self._cross_asset_lag.detect_lag_signal(asset)
            if _lag_signal and _lag_signal["confidence"] > confidence:
                direction = _lag_signal["direction"]
                confidence = _lag_signal["confidence"]
                primary = "CROSS_ASSET_LAG"
                diag["cross_asset_lag"] = _lag_signal["meta"]
        except Exception as _lag_err:
            logger.debug(f"[UPG5] Cross-asset lag check failed: {_lag_err}")

        return {
            "lead_lag_edge": round(lead_lag_edge, 2),
            "lead_lag_confidence": round(lead_lag_conf, 4),
            "cvd_divergence": round(cvd_div, 4),
            "order_book_imbalance": round(ob_imb, 4),
            "spread_bps": round(spread, 2),
            "micro_direction": round(direction, 4),
            "micro_confidence": round(confidence, 4),
            "micro_urgency": round(urgency, 4),
            "micro_primary_signal": primary,
            "micro_data_quality": round(quality, 3),
            "micro_is_valid": True,
            "data_age_seconds": round(age_ms / 1000, 3),
            "asof_timestamp": now,
            "asset": asset,
            "micro_cross_asset_lag": _lag_signal,  # [UPG5] None or signal dict
            "diagnostics": diag,
        }

    # ---- v6 contract aliases ------------------------------------------------

    def to_agent_signal_dict(self, asset: str = "BTC") -> Dict[str, Any]:
        """
        Alias for generate_signal() - v6 agent contract.

        Returns the last cached payload if the asset matches, otherwise
        generates a fresh signal.
        """
        st = self._per_asset.get(asset)
        if st and st.last_v6_payload and st.last_v6_payload.get("asset") == asset:
            return st.last_v6_payload
        return self.generate_signal(asset=asset)

    def to_market_data_patch(self, asset: str = "BTC") -> Dict[str, Any]:
        """
        Return canonical microstructure keys for market_data enrichment.

        These keys are written *into* the market_data dict consumed by
        constitution / trade-gate / execution - they are NOT authority
        signals.

        Keys produced (all Optional[float], None = unknown):
          spread_bps, order_book_imbalance, lead_lag_edge,
          cvd_divergence, micro_data_age_seconds
        """
        sig = self.to_agent_signal_dict(asset)
        return {
            "spread_bps": sig.get("spread_bps"),
            "order_book_imbalance": sig.get("order_book_imbalance"),
            "lead_lag_edge": sig.get("lead_lag_edge"),
            "cvd_divergence": sig.get("cvd_divergence"),
            "micro_data_age_seconds": sig.get("data_age_seconds"),
        }

    # Alias for spec compliance
    get_market_data_patch = to_market_data_patch

    def analyze_asset(
        self,
        asset: str,
        market_data: Optional[Dict] = None,
        now_ts: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        V6.4 execution-focused analysis - direction-neutral.

        Returns execution-relevant microstructure fields ONLY:
          spread_bps (float), order_book_imbalance (float in [-1,1]),
          orderbook_depth (float ratio/proxy), estimated_friction_bps (float),
          flash_crash_active (bool), price_move_pct (float),
          data_quality (str: ok/stale/missing/error),
          asof_ts (str ISO), data_age_seconds (float),
          provenance (list[str]).
        """
        try:
            sig = self.generate_signal(asset=asset, market_data=market_data)

            spread = sig.get("spread_bps", 0.0) or 0.0
            ob_imb = sig.get("order_book_imbalance", 0.0) or 0.0
            age_s = sig.get("data_age_seconds", 0.0) or 0.0
            quality_f = sig.get("micro_data_quality", 0.0) or 0.0

            # Determine data_quality string
            if not sig.get("micro_is_valid", True):
                dq = "error"
            elif quality_f < 0.1:
                dq = "missing"
            elif age_s > self.config.max_snapshot_age_ms / 1000:
                dq = "stale"
            else:
                dq = "ok"

            # Execution heuristics
            friction = spread / 2.0  # half-spread + no impact proxy (conservative)
            price_move = 0.0
            if market_data:
                pm = market_data.get("price_move_pct")
                if pm is not None:
                    price_move = float(pm)
            flash_crash = abs(price_move) > 8.0

            # orderbook_depth proxy: based on ob_imb extremeness
            # 0 = balanced, 1 = extreme one-sided
            ob_depth_proxy = max(0.0, 1.0 - abs(ob_imb))

            return {
                "spread_bps": round(spread, 2),
                "order_book_imbalance": round(ob_imb, 4),
                "orderbook_depth": round(ob_depth_proxy, 4),
                "estimated_friction_bps": round(friction, 2),
                "flash_crash_active": flash_crash,
                "price_move_pct": round(price_move, 4),
                "data_quality": dq,
                "asof_ts": _iso_utc(now_ts),
                "data_age_seconds": round(age_s, 3),
                "provenance": ["microstructure_agent", asset],
            }
        except Exception as exc:
            logger.error("[MicroV6] analyze_asset(%s) failed: %s", asset, exc)
            return {
                "spread_bps": 0.0,
                "order_book_imbalance": 0.0,
                "orderbook_depth": 0.0,
                "estimated_friction_bps": 0.0,
                "flash_crash_active": False,
                "price_move_pct": 0.0,
                "data_quality": "error",
                "asof_ts": _iso_utc(now_ts),
                "data_age_seconds": 0.0,
                "provenance": ["microstructure_agent", asset, f"error:{exc}"],
            }

    # ---- convenience accessors ---------------------------------------------

    def get_last_payload(self, asset: str) -> Dict[str, Any]:
        """Return last cached v6 payload (or neutral)."""
        st = self._per_asset.get(asset)
        if st and st.last_v6_payload:
            return st.last_v6_payload
        return _neutral_v6_payload(asset)

    def get_microstructure_bias(self, asset: str = "BTC") -> Tuple[float, float]:
        """Get overall microstructure bias as (direction, confidence)."""
        st = self._per_asset.get(asset)
        if st is None:
            return 0.0, 0.0

        biases = []

        flow_imb = st.taker_analyzer.get_imbalance()
        if abs(flow_imb) > 0.1:
            biases.append((flow_imb, 0.5))

        if len(st.imbalance_history) > 5:
            book_imb = float(np.mean(list(st.imbalance_history)[-5:]))
            if abs(book_imb) > 0.1:
                biases.append((book_imb, 0.4))

        if not biases:
            return 0.0, 0.0

        total_w = sum(w for _, w in biases)
        weighted = sum(b * w for b, w in biases) / total_w
        avg_conf = total_w / len(biases)
        return weighted, avg_conf

    def get_status(self, asset: str = None) -> Dict:
        """Agent status for one asset or all."""
        if asset:
            st = self._per_asset.get(asset)
            if st is None:
                return {"asset": asset, "state": "no_data"}
            return {
                "asset": asset,
                "exchanges_tracked": list(st.exchange_states.keys()),
                "spread_history_len": len(st.spread_history),
                "imbalance_history_len": len(st.imbalance_history),
                "lag_estimates": st.lag_detector.lag_estimates,
                "cross_spread_bps": (st.cross_state.cross_spread_bps
                                     if st.cross_state else 0),
                "taker_imbalance": st.taker_analyzer.get_imbalance(),
                "signals_generated": len(st.signals_generated),
            }
        return {a: self.get_status(a) for a in self._per_asset}

    def reset_state(self, asset: str = None):
        """Clear state for one asset or all."""
        if asset:
            self._per_asset.pop(asset, None)
        else:
            self._per_asset.clear()


# =============================================================================
# EVENTBUS INTEGRATION HELPER
# =============================================================================

class MicrostructureEventBusIntegration:
    """
    Optional EventBus wiring for the MicrostructureArbitrageAgent.

    Subscribes to LOB_TICK / TRADE_FLOW events, publishes ALPHA_SIGNAL.
    Does NOT start any background loops.
    """

    def __init__(self, agent: MicrostructureArbitrageAgent):
        self.agent = agent
        self._event_manager = None
        self._subscribed = False

    def initialize(self) -> bool:
        """Wire up to the unified event manager.  Returns False on failure."""
        try:
            from infra.unified_event_v521 import (
                get_unified_event_manager,
                EventTypeV521,
            )
            self._event_manager = get_unified_event_manager()

            self._event_manager.subscribe(
                EventTypeV521.LOB_TICK,
                self._on_lob_tick,
            )
            self._event_manager.subscribe(
                EventTypeV521.TRADE_FLOW,
                self._on_trade_flow,
            )

            self._subscribed = True
            logger.info("MicrostructureEventBusIntegration initialised")
            return True

        except Exception as e:
            logger.warning("Microstructure EventBus integration failed: %s", e)
            return False

    def _on_lob_tick(self, event):
        """Handle LOB_TICK: extract bid/ask and feed into agent."""
        payload = getattr(event, "payload", {}) or {}
        asset = payload.get("asset", "BTC")
        exchange = payload.get("exchange", "kraken")
        bid = payload.get("bid", 0)
        ask = payload.get("ask", 0)
        if bid > 0 and ask > 0:
            self.agent.update_exchange(
                asset=asset,
                exchange=exchange,
                bid=bid,
                ask=ask,
                bid_size=payload.get("bid_size", 1.0),
                ask_size=payload.get("ask_size", 1.0),
                timestamp_ms=payload.get("timestamp_ms"),
            )

    def _on_trade_flow(self, event):
        """Handle TRADE_FLOW: feed taker trades into agent."""
        payload = getattr(event, "payload", {}) or {}
        asset = payload.get("asset", "BTC")
        exchange = payload.get("exchange", "kraken")
        side = payload.get("side", "")
        size = payload.get("size", 0)
        if side and size > 0:
            st = self.agent._state(asset)
            ts_ms = payload.get("timestamp_ms", time.time() * 1000)
            st.taker_analyzer.add_trade(side, size, ts_ms)

    def publish_signal(self, asset: str):
        """Generate signal and publish as ALPHA_SIGNAL event."""
        if not self._event_manager:
            return
        try:
            from infra.unified_event_v521 import UnifiedEvent, EventTypeV521

            payload = self.agent.generate_signal(asset)
            event = UnifiedEvent(
                event_type=EventTypeV521.ALPHA_SIGNAL,
                source="microstructure_agent",
                payload=payload,
            )
            self._event_manager.process_event(event)
        except Exception as e:
            logger.warning("Failed to publish microstructure signal: %s", e)


# =============================================================================
# SINGLETON
# =============================================================================

_microstructure_agent: Optional[MicrostructureArbitrageAgent] = None


def get_microstructure_agent(
    config: MicrostructureConfig = None,
) -> MicrostructureArbitrageAgent:
    """Get or create microstructure agent singleton."""
    global _microstructure_agent
    if _microstructure_agent is None:
        _microstructure_agent = MicrostructureArbitrageAgent(config)
    elif config is not None and _microstructure_agent.config != config:
        logger.info("MicrostructureArbitrageAgent config changed; refreshing singleton instance")
        _microstructure_agent = MicrostructureArbitrageAgent(config)
    return _microstructure_agent


def reset_microstructure_agent():
    """Reset microstructure agent singleton."""
    global _microstructure_agent
    _microstructure_agent = None


def _iso_utc(ts=None):
    """[P323b] ISO-8601 UTC with a SINGLE "Z" marker.

    `isoformat()` on an aware datetime already emits "+00:00", so the old
    `isoformat() + "Z"` produced "+00:00Z" — which broke /health's freshness
    parse for four months (P323). A NAIVE value is normalised to UTC first
    rather than merely reformatted: naive + "Z" labels local time as UTC
    (P40/P97), which is wrong, not just malformed.
    """
    from datetime import datetime as _dt, timezone as _tz
    t = ts or _dt.now(_tz.utc)
    if t.tzinfo is None:
        t = t.replace(tzinfo=_tz.utc)
    return t.isoformat().replace("+00:00", "Z")

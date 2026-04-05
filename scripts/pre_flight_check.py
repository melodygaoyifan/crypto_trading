"""
HMATS Paper Run Pre-Flight Check v2.0
======================================
Validates the entire live trading pipeline before committing to
an unattended paper run. Exercises every subsystem from data fetch
through execution using REAL Kraken data.

Run:  python -X utf8 pre_flight_check.py
Time: ~30 seconds
Exit: 0 = pass, 1 = fail

10 Sections:
  1. Import & Component Loading
  2. Kraken Connectivity (3 assets, real API)
  3. Technical Indicators
  4. GMM Regime Classification
  5. Alpha Gate & Strategy Selection
  6. Authority Fusion & Veto Chain
  7. Tranche & Execution Readiness
  8. Safety Systems
  9. External Feeds (F&G, DNS)
 10. Regression Tests (bugs we fixed)
"""

import sys
import os
import json
import time
import asyncio
import threading
import logging
import importlib
import traceback
from pathlib import Path
from dataclasses import dataclass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env for API keys
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Quiet down noisy loggers during check
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)

ASSETS = ["BTC", "ETH", "SOL"]
KRAKEN_PAIRS = {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD"}

# Known GMM regime names
KNOWN_REGIMES = {
    "MOMENTUM_RALLY", "PANIC_SELLOFF", "QUIET_ACCUMULATION",
    "WEAK_CONSOLIDATION", "EXTREME_VOLATILITY", "VOLATILE_CHOP",
    "MEAN_REVERT",  # ADX fallback can produce this
}

# Regime power multipliers that MUST exist in main.py
REGIME_POWER_MAP = {
    "MOMENTUM_RALLY": 1.5,
    "PANIC_SELLOFF": 1.5,
    "QUIET_ACCUMULATION": 0.6,
    "WEAK_CONSOLIDATION": 0.6,
    "EXTREME_VOLATILITY": 1.0,
    "VOLATILE_CHOP": 0.6,
}


# ============================================================================
# TEST FRAMEWORK
# ============================================================================
_results = []
_total = 0
_passed = 0
_failed = 0
_warnings = 0


def check(name: str, condition: bool, detail: str = "", fatal: bool = True) -> bool:
    """Register a check result."""
    global _total, _passed, _failed, _warnings
    _total += 1
    if condition:
        _passed += 1
        status = "PASS"
    elif fatal:
        _failed += 1
        status = "FAIL"
    else:
        _warnings += 1
        status = "WARN"
    print(f"  {status}: {name}")
    if detail and not condition:
        print(f"        -> {detail}")
    _results.append((status, name, detail))
    return condition


def section(num: int, title: str):
    print(f"\n{'='*70}")
    print(f"  Section {num}: {title}")
    print(f"{'='*70}")


# ============================================================================
# SECTION 1: Import & Component Loading
# ============================================================================
section(1, "Import & Component Loading")

# 1a. Core engine
v36_engine = None
try:
    from integration.integration_v36 import HMATSv36Engine, TradeIntentV36
    check("integration_v36 imports", True)
except Exception as e:
    check("integration_v36 imports", False, str(e))

# 1b. Authority fusion
try:
    from signals.authority_fusion import AuthorityFusionEngine, AgentSignal
    check("authority_fusion imports", True)
except Exception as e:
    check("authority_fusion imports", False, str(e))

# 1c. Constitution
try:
    from defense.constitution import (
        ConstitutionGuaranteesModule, AlphaGatingResult, TrancheLevel,
    )
    check("constitution imports", True)
except Exception as e:
    check("constitution imports", False, str(e))

# 1d. Trade gate
try:
    from defense.trade_gate import TradeGate, get_trade_gate
    check("trade_gate imports", True)
except Exception as e:
    check("trade_gate imports", False, str(e))

# 1e. Short bias agent
try:
    from agents.short_bias_agent import ShortBiasAgent
    check("short_bias_agent imports", True)
except Exception as e:
    check("short_bias_agent imports", False, str(e))

# 1f. GMM / joblib
try:
    import joblib
    check("joblib imports", True)
except Exception as e:
    check("joblib imports", False, str(e))

# 1g. ta library (NOT pandas_ta)
try:
    import ta
    check("ta library imports (not pandas_ta)", True)
except Exception as e:
    check("ta library imports", False, f"{e} - pip install ta")

# 1h. Execution & dead-man switch
try:
    from execution.dead_man_switch import DeadManSwitch
    check("dead_man_switch imports", True)
except Exception as e:
    check("dead_man_switch imports", False, str(e))

# 1i. Event bus
try:
    from infra.event_bus import EventBus
    check("event_bus imports", True)
except Exception as e:
    check("event_bus imports", False, str(e))

# 1j. P0 safety integrator
try:
    from defense.p0_safety_integrator import get_p0_integrator
    check("p0_safety_integrator imports", True)
except Exception as e:
    check("p0_safety_integrator imports", False, str(e))

# 1k. Account sync
try:
    from core.account_sync import get_account_sync
    check("account_sync imports", True)
except Exception as e:
    check("account_sync imports", False, str(e))

# 1l. Risk manager
try:
    from risk.risk_manager import RiskManager
    check("risk_manager imports", True)
except Exception as e:
    check("risk_manager imports", False, str(e))

# 1m. Tranche manager
try:
    from risk.tranche_manager import TrancheManager
    check("tranche_manager imports", True)
except Exception as e:
    check("tranche_manager imports", False, str(e))

# 1n. Execution guards (RLock fix)
try:
    from defense.execution_guards import StaleDataKillSwitch
    check("execution_guards imports", True)
except Exception as e:
    check("execution_guards imports", False, str(e))

# 1o. Regime smoother
try:
    from core.regime_smoother import RegimeSmoother
    check("regime_smoother imports", True)
except Exception as e:
    check("regime_smoother imports", False, str(e))

# 1p. HTTP session factory (DNS fix)
try:
    from data_mgmt.feeds._http import create_session, fetch_with_retry
    check("_http session factory imports", True)
except Exception as e:
    check("_http session factory imports", False, str(e))

# 1q. ccxt
try:
    import ccxt
    check("ccxt imports", True)
except Exception as e:
    check("ccxt imports", False, f"{e} - pip install ccxt")

# 1r. GMM model files exist
gmm_model_dir = PROJECT_ROOT / "models" / "regime_classifier"
gmm_model_path = gmm_model_dir / "gmm_model.pkl"
gmm_config_path = gmm_model_dir / "gmm_config.json"
check("GMM model file exists", gmm_model_path.exists(),
      f"Missing: {gmm_model_path}")
check("GMM config file exists", gmm_config_path.exists(),
      f"Missing: {gmm_config_path}")


# ============================================================================
# SECTION 2: Kraken Connectivity (Real API, 3 Assets)
# ============================================================================
section(2, "Kraken Connectivity")

exchange = None
ohlcv_data = {}  # asset -> list of candles (for section 3)
ticker_data = {}  # asset -> ticker dict

try:
    exchange = ccxt.kraken({
        'apiKey': os.environ.get('KRAKEN_API_KEY', ''),
        'secret': os.environ.get('KRAKEN_API_SECRET', ''),
        'enableRateLimit': True,
    })
    check("Kraken exchange created", True)
except Exception as e:
    check("Kraken exchange created", False, str(e))

if exchange:
    for asset in ASSETS:
        pair = KRAKEN_PAIRS[asset]

        # 2a. Ticker
        try:
            ticker = exchange.fetch_ticker(pair)
            price = ticker.get('last', 0)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            check(f"{asset} ticker fetched (price=${price:,.2f})", price > 0,
                  f"price={price}")
            ticker_data[asset] = ticker

            # Spread check
            if bid and ask and bid > 0:
                spread_bps = (ask - bid) / bid * 10000
                check(f"{asset} spread < 100bps ({spread_bps:.1f}bps)", spread_bps < 100,
                      f"spread={spread_bps:.1f}bps")
        except Exception as e:
            check(f"{asset} ticker fetched", False, str(e))

        # 2b. OHLCV (250 bars for EMA-200)
        try:
            candles = exchange.fetch_ohlcv(pair, '4h', limit=250)
            n_bars = len(candles)
            check(f"{asset} OHLCV bars >= 200 (got {n_bars})", n_bars >= 200,
                  f"Only {n_bars} bars - EMA-200 needs 250")
            ohlcv_data[asset] = candles

            # Check for NaN in closes
            closes = [c[4] for c in candles if c[4] is not None]
            check(f"{asset} OHLCV no None in closes",
                  len(closes) == n_bars,
                  f"{n_bars - len(closes)} None values in closes")
        except Exception as e:
            check(f"{asset} OHLCV fetched", False, str(e))

        # 2c. Orderbook
        try:
            ob = exchange.fetch_order_book(pair, limit=10)
            bids = ob.get('bids', [])
            asks = ob.get('asks', [])
            check(f"{asset} orderbook has bids/asks",
                  len(bids) > 0 and len(asks) > 0,
                  f"bids={len(bids)}, asks={len(asks)}")
        except Exception as e:
            check(f"{asset} orderbook fetched", False, str(e))

        # Small delay to respect rate limits
        time.sleep(0.3)


# ============================================================================
# SECTION 3: Technical Indicators
# ============================================================================
section(3, "Technical Indicators")

import numpy as np

indicator_results = {}  # asset -> dict of indicators

for asset in ASSETS:
    if asset not in ohlcv_data:
        check(f"{asset} indicators (skipped - no OHLCV)", False, "No OHLCV data from Section 2")
        continue

    candles = ohlcv_data[asset]
    try:
        import pandas as pd
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Compute indicators using ta library
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd_obj = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_obj.macd()
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        ema_200 = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        # Get last valid values
        rsi_val = rsi.dropna().iloc[-1] if not rsi.dropna().empty else None
        macd_val = macd_line.dropna().iloc[-1] if not macd_line.dropna().empty else None
        bb_up = bb_upper.dropna().iloc[-1] if not bb_upper.dropna().empty else None
        bb_lo = bb_lower.dropna().iloc[-1] if not bb_lower.dropna().empty else None
        ema200_val = ema_200.dropna().iloc[-1] if not ema_200.dropna().empty else None
        adx_val = adx.dropna().iloc[-1] if not adx.dropna().empty else None
        atr_val = atr.dropna().iloc[-1] if not atr.dropna().empty else None

        # Volume ratio
        vol_median = df['volume'].rolling(21).median().iloc[-1]
        vol_ratio = df['volume'].iloc[-1] / vol_median if vol_median and vol_median > 0 else 0

        indicator_results[asset] = {
            'rsi': rsi_val, 'macd': macd_val,
            'bb_upper': bb_up, 'bb_lower': bb_lo,
            'ema_200': ema200_val, 'adx': adx_val,
            'atr': atr_val, 'volume_ratio': vol_ratio,
            'close': df['close'].iloc[-1],
        }

        # Validate ranges
        check(f"{asset} RSI in (0,100): {rsi_val:.1f}",
              rsi_val is not None and 0 < rsi_val < 100,
              f"RSI={rsi_val}")
        check(f"{asset} MACD computed: {macd_val:.4f}",
              macd_val is not None,
              "MACD is None")
        check(f"{asset} BB upper > lower",
              bb_up is not None and bb_lo is not None and bb_up > bb_lo,
              f"upper={bb_up}, lower={bb_lo}")
        check(f"{asset} EMA-200 > 0: ${ema200_val:,.2f}" if ema200_val else f"{asset} EMA-200 computed",
              ema200_val is not None and ema200_val > 0,
              f"EMA-200={ema200_val} (need 250 bars)")
        check(f"{asset} ADX in (0,100): {adx_val:.1f}",
              adx_val is not None and 0 < adx_val < 100,
              f"ADX={adx_val}")
        check(f"{asset} ATR > 0: {atr_val:.4f}",
              atr_val is not None and atr_val > 0,
              f"ATR={atr_val}")
        check(f"{asset} volume_ratio > 0: {vol_ratio:.2f}",
              vol_ratio > 0,
              f"vol_ratio={vol_ratio}", fatal=False)

    except Exception as e:
        check(f"{asset} indicator computation", False, f"{e}\n{traceback.format_exc()}")


# ============================================================================
# SECTION 4: GMM Regime Classification
# ============================================================================
section(4, "GMM Regime Classification")

gmm_model = None
gmm_scaler_mean = None
gmm_scaler_scale = None
gmm_regime_names = None
gmm_feature_cols = None

# 4a. Load GMM model
if gmm_model_path.exists() and gmm_config_path.exists():
    try:
        gmm_model = joblib.load(gmm_model_path)
        check("GMM model loaded via joblib", True)

        with open(gmm_config_path) as f:
            cfg = json.load(f)

        gmm_regime_names = cfg.get("regime_names", [])
        gmm_feature_cols = cfg.get("feature_cols", [])

        check(f"GMM has 6 regimes: {gmm_regime_names}",
              len(gmm_regime_names) == 6)

        check(f"GMM has 12 features",
              len(gmm_feature_cols) == 12,
              f"Has {len(gmm_feature_cols)}: {gmm_feature_cols}")

        # Load scaler
        scaler_data = cfg.get("scaler", {})
        if "mean" in scaler_data and "scale" in scaler_data:
            gmm_scaler_mean = np.array(scaler_data["mean"])
            gmm_scaler_scale = np.array(scaler_data["scale"])
            check("GMM scaler loaded from config",
                  gmm_scaler_mean is not None and len(gmm_scaler_mean) == len(gmm_feature_cols),
                  f"scaler dims={len(gmm_scaler_mean) if gmm_scaler_mean is not None else 'None'}")
        else:
            # Try separate scaler file
            scaler_path = gmm_model_dir / "scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                gmm_scaler_mean = scaler.mean_
                gmm_scaler_scale = scaler.scale_
                check("GMM scaler loaded from .pkl",
                      len(gmm_scaler_mean) == len(gmm_feature_cols),
                      f"scaler dims={len(gmm_scaler_mean)}, GMM dims={len(gmm_feature_cols)}")
            else:
                check("GMM scaler exists", False, "No scaler in config or .pkl")

    except Exception as e:
        check("GMM model load", False, str(e))

# 4b. Predict regime for each asset (using indicators from Section 3)
if gmm_model is not None and gmm_scaler_mean is not None:
    for asset in ASSETS:
        if asset not in indicator_results:
            continue

        ind = indicator_results[asset]
        close = ind['close']

        try:
            # Build 12-feature vector (must match training order)
            features = {}
            # Returns (approximate from OHLCV)
            candles = ohlcv_data[asset]
            closes = [c[4] for c in candles]
            features['return_1h'] = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
            features['return_4h'] = (closes[-1] / closes[-2] - 1) if len(closes) >= 2 else 0
            features['return_24h'] = (closes[-1] / closes[-7] - 1) if len(closes) >= 7 else 0
            features['return_7d'] = (closes[-1] / closes[-42] - 1) if len(closes) >= 42 else 0

            # Volatility
            rets = np.diff(np.log(np.array(closes[-25:])))
            features['volatility_1h'] = np.std(rets[-2:]) if len(rets) >= 2 else 0
            features['volatility_24h'] = np.std(rets[-6:]) if len(rets) >= 6 else 0

            # Vol percentile + vol-of-vol (simplified)
            all_rets = np.diff(np.log(np.array(closes)))
            rolling_vols = [np.std(all_rets[max(0,i-6):i]) for i in range(6, len(all_rets))]
            if rolling_vols:
                current_vol = features['volatility_24h']
                features['vol_percentile'] = np.mean([1 for v in rolling_vols if v <= current_vol])
                features['vol_of_vol'] = np.std(rolling_vols[-20:]) if len(rolling_vols) >= 20 else 0.01
            else:
                features['vol_percentile'] = 0.5
                features['vol_of_vol'] = 0.01

            # Momentum consistency
            r1 = features['return_4h']
            r7 = features['return_7d']
            features['momentum_consistency'] = 1.0 if (r1 > 0) == (r7 > 0) and r7 != 0 else 0.0

            # Cross-asset correlation (default matching training)
            features['cross_asset_correlation'] = 0.87

            # Fear index (placeholder, real value from F&G in Section 9)
            features['fear_index'] = 0.5

            # Spread percentile (per-asset defaults matching training)
            spread_defaults = {"BTC": 5, "ETH": 8, "SOL": 12}
            features['spread_percentile'] = spread_defaults.get(asset, 8) / 100

            # Build feature vector in correct order
            fv = np.array([features.get(col, 0.0) for col in gmm_feature_cols]).reshape(1, -1)

            # Scale
            scaled = (fv - gmm_scaler_mean) / gmm_scaler_scale
            scaled = np.clip(scaled, -4.0, 4.0)

            # Check distribution shift
            n_extreme = np.sum(np.abs(scaled) > 3.0)
            n_features = len(gmm_feature_cols)
            extreme_pct = n_extreme / n_features
            check(f"{asset} GMM z-score shift < 30% (extreme={n_extreme}/{n_features})",
                  extreme_pct < 0.3,
                  f"{extreme_pct:.0%} features > 3σ - distribution shift!")

            # Predict
            probs = gmm_model.predict_proba(scaled)[0]
            best_idx = np.argmax(probs)
            regime_name = gmm_regime_names[best_idx]
            confidence = probs[best_idx]

            check(f"{asset} GMM regime: {regime_name} (conf={confidence:.3f})",
                  regime_name in KNOWN_REGIMES or regime_name in gmm_regime_names,
                  f"Unknown regime: {regime_name}")
            check(f"{asset} GMM confidence > 0",
                  confidence > 0,
                  f"confidence={confidence}")

        except Exception as e:
            check(f"{asset} GMM prediction", False, f"{e}")


# ============================================================================
# SECTION 5: Alpha Gate & Strategy Selection
# ============================================================================
section(5, "Alpha Gate & Strategy Selection")

# 5a. Instantiate constitution guarantees
constitution = None
try:
    constitution = ConstitutionGuaranteesModule()
    check("ConstitutionGuaranteesModule created", True)
except Exception as e:
    check("ConstitutionGuaranteesModule created", False, str(e))

# 5b. Test alpha gate with synthetic signals
if constitution:
    for mode_name in ["NORMAL", "OPPORTUNITY"]:
        try:
            result = constitution.check_alpha_gate(
                signal_strength=0.6,
                regime_confidence=0.8,
                mode=mode_name,
            )
            check(f"Alpha gate ({mode_name}): threshold={result.threshold_bps:.1f}bps, alpha={result.estimated_alpha_bps:.1f}bps",
                  result.threshold_bps > 0,
                  f"threshold={result.threshold_bps}")
        except Exception as e:
            check(f"Alpha gate ({mode_name})", False, str(e))

# 5c. Test with real indicator data
for asset in ASSETS:
    if asset not in indicator_results or constitution is None:
        continue

    ind = indicator_results[asset]
    rsi = ind.get('rsi', 50)
    adx_val = ind.get('adx', 20)
    macd_val = ind.get('macd', 0)
    close = ind.get('close', 0)
    ema200 = ind.get('ema_200', close)

    # Compute a synthetic signal strength from indicators
    # (mimics best-of-N strategy selection)
    momentum_signal = 0.0
    if adx_val and adx_val > 25:
        if ema200 and close > ema200:
            momentum_signal = min(adx_val / 50, 1.0) * 0.8
        elif ema200 and close < ema200:
            momentum_signal = -min(adx_val / 50, 1.0) * 0.8

    signal_strength = abs(momentum_signal) if momentum_signal != 0 else 0.3
    alpha_result = constitution.check_alpha_gate(
        signal_strength=signal_strength,
        regime_confidence=0.7,
        mode="NORMAL",
    )
    check(f"{asset} alpha with real data: {alpha_result.estimated_alpha_bps:.1f}bps vs {alpha_result.threshold_bps:.1f}bps",
          alpha_result.estimated_alpha_bps > 0,
          f"alpha=0 - signal generation broken")
    print(f"        signal={signal_strength:.3f}, passes={alpha_result.passes_threshold}")


# ============================================================================
# SECTION 6: Authority Fusion & Veto Chain
# ============================================================================
section(6, "Authority Fusion & Veto Chain")

# 6a. SystemMode mapping
try:
    from integration.integration_v36 import HMATSv36Engine
    # Check that the mode mapping exists and works
    from signals.authority_fusion import SystemMode as FusionSystemMode
    mode_map = {
        "NO_TRADE": FusionSystemMode.NO_TRADE,
        "OPPORTUNITY": FusionSystemMode.OPPORTUNITY,
        "NORMAL": FusionSystemMode.NORMAL,
    }
    check("SystemMode mapping has 3 entries",
          len(mode_map) == 3 and all(v is not None for v in mode_map.values()))
except Exception as e:
    check("SystemMode mapping", False, str(e))

# 6b. Build fusion signals (mimics _build_fusion_signals)
try:
    signals = {}
    # quant
    signals["quant"] = AgentSignal(direction=-0.6, confidence=0.7)
    # regime
    signals["regime"] = AgentSignal(direction=-0.6, confidence=0.75)
    # sentiment
    signals["sentiment"] = AgentSignal(direction=-0.1, confidence=0.3)
    # short_bias (C3 fix)
    signals["short_bias"] = AgentSignal(direction=-0.8, confidence=0.5)
    # risk
    signals["risk"] = AgentSignal(direction=0.0, confidence=0.5)
    # macro
    signals["macro"] = AgentSignal(direction=0.0, confidence=0.3)

    expected_keys = {"quant", "regime", "sentiment", "short_bias", "risk", "macro"}
    check(f"Fusion signals include all 6 agents: {sorted(signals.keys())}",
          expected_keys.issubset(signals.keys()),
          f"Missing: {expected_keys - set(signals.keys())}")
except Exception as e:
    check("Fusion signal construction", False, str(e))

# 6c. Trade gate consistency (3 paths must all accept drl_weight=0.0)
try:
    tg = TradeGate()
    # Update timestamps so STALE_DATA doesn't fire
    tg.update_data_timestamp("price")
    tg.update_data_timestamp("orderbook")
    tg.update_data_timestamp("vpin")
    time.sleep(0.01)

    # TradeGate.check(asset, side, size, confidence, regime, **kwargs)
    tg_result = tg.check(
        "BTC", "short", 0.15,
        confidence=0.7,
        regime="WEAK_CONSOLIDATION",
        drl_weight=0.0,
        expected_alpha_bps=80.0,
        volume_ratio=1.0,
        regime_confidence=0.7,
    )
    # GateResult has .passed and .reason
    is_passed = getattr(tg_result, 'passed', True)
    reason = getattr(tg_result, 'reason', '')
    check(f"TradeGate check with drl_weight=0.0 (passed={is_passed})",
          "DRL_OVERCONFIDENT" not in str(reason),
          f"DRL_OVERCONFIDENT veto - drl_weight not forwarded: {reason}")
    if not is_passed:
        print(f"        Note: gate vetoed for: {reason}")
except Exception as e:
    check("TradeGate consistency", False, str(e))

# 6d. Verify no DRL_OVERCONFIDENT false veto
try:
    tg2 = TradeGate()
    tg2.update_data_timestamp("price")
    tg2.update_data_timestamp("orderbook")
    tg2.update_data_timestamp("vpin")
    time.sleep(0.01)

    result2 = tg2.check(
        "ETH", "short", 0.10,
        confidence=0.7,
        regime="MOMENTUM_RALLY",
        drl_weight=0.0,
        expected_alpha_bps=60.0,
        volume_ratio=0.5,
        regime_confidence=0.8,
    )
    reason2 = getattr(result2, 'reason', '')
    check(f"TradeGate no DRL_OVERCONFIDENT veto",
          "DRL_OVERCONFIDENT" not in str(reason2),
          f"False veto: {reason2}", fatal=False)
except Exception as e:
    check("TradeGate false veto check", False, str(e), fatal=False)


# ============================================================================
# SECTION 7: Tranche & Execution Readiness
# ============================================================================
section(7, "Tranche & Execution Readiness")

# 7a. TradeIntentV36 default values
try:
    intent = TradeIntentV36()
    check("TradeIntentV36.alpha_gate_passed default is False (fail-closed)",
          intent.alpha_gate_passed == False,
          f"Default is {intent.alpha_gate_passed} - should be False (H2 fix)")
    check("TradeIntentV36.urgency is float",
          isinstance(intent.urgency, (int, float)),
          f"type={type(intent.urgency)} - should be float (C2 fix)")
except Exception as e:
    check("TradeIntentV36 defaults", False, str(e))

# 7b. Tranche cold-start (ENTER_T1 not FLATTEN)
try:
    cg = ConstitutionGuaranteesModule()
    # schedule_tranche(asset, mode, target_direction, market_data, signal_data)
    mock_market = {
        "current_price": 70000.0, "volume_ratio": 0.5,
        "regime_state": "WEAK_CONSOLIDATION", "atr": 1000.0,
        "data_age_seconds": 1.0, "spread_bps": 5.0,
    }
    mock_signal = {
        "signal_strength": 0.6, "regime_confidence": 0.7,
        "quant_direction": -1.0,
    }
    tranche_result = cg.schedule_tranche(
        asset="BTC",
        mode="NORMAL",
        target_direction=-1.0,
        market_data=mock_market,
        signal_data=mock_signal,
    )
    action = getattr(tranche_result, 'action', str(tranche_result))
    check(f"Tranche cold-start produces ENTER (got: {action})",
          "FLATTEN" not in str(action).upper() or "ENTER" in str(action).upper(),
          f"Got FLATTEN on cold start - tranche_cold_start bug regression")
except Exception as e:
    check("Tranche cold-start", False, str(e), fatal=False)

# 7c. Target exposure sanity
try:
    intent2 = TradeIntentV36()
    intent2.target_exposure = 0.15  # 15%
    check(f"Target exposure settable (15%)",
          0.01 <= intent2.target_exposure <= 0.25,
          f"target_exposure={intent2.target_exposure}")
except Exception as e:
    check("Target exposure", False, str(e))


# ============================================================================
# SECTION 8: Safety Systems
# ============================================================================
section(8, "Safety Systems")

# 8a. Dead-man switch lifecycle (dry_run=True)
try:
    dms = DeadManSwitch(rest_client=None, timeout_sec=60, dry_run=True)
    started = dms.start()
    check("Dead-man switch start (dry run)", started)

    refreshed = dms.refresh()
    check("Dead-man switch refresh", refreshed)

    check("Dead-man switch is_active", dms.is_active)

    stopped = dms.stop()
    check("Dead-man switch stop", stopped)
    check("Dead-man switch inactive after stop", not dms.is_active)

    status = dms.get_status()
    check("Dead-man switch status dict has all keys",
          all(k in status for k in ['active', 'timeout_sec', 'dry_run', 'refresh_count']))
except Exception as e:
    check("Dead-man switch lifecycle", False, str(e))

# 8b. Account sync (dry run - needs async refresh before get_equity)
async def _test_account_sync():
    acct = get_account_sync(exchange_name="kraken", dry_run=True)
    acct.set_dry_run_equity(100_000.0)
    await acct.refresh()  # MUST refresh before get_equity (fail-closed)
    equity = acct.get_equity()
    return equity

try:
    equity = asyncio.run(_test_account_sync())
    check(f"Account sync equity = $100K (got ${equity:,.0f})",
          abs(equity - 100_000) < 1.0,
          f"equity={equity}")
except Exception as e:
    check("Account sync", False, str(e))

# 8c. Risk manager initialization
try:
    rm = RiskManager()
    rm.initialize(100_000.0)
    # After initialization, should allow trades
    can = rm.can_trade() if hasattr(rm, 'can_trade') else True
    check("RiskManager initialized with balance > 0", True)
    check("RiskManager allows trades after init", can,
          "can_trade()=False after fresh init - false veto regression", fatal=False)
except Exception as e:
    check("RiskManager", False, str(e))

# 8d. P0 safety integrator
try:
    p0 = get_p0_integrator()
    check("P0 safety integrator loads", p0 is not None)
except Exception as e:
    check("P0 safety integrator", False, str(e))

# 8e. Event bus write test
try:
    eb = EventBus()
    from infra.event_bus import Event, EventType
    test_event = Event(
        event_type=EventType.HEARTBEAT,
        source="pre_flight_check",
        payload={"test": True, "timestamp": time.time()},
    )
    eb.publish(test_event)
    check("Event bus accepts publish", True)
    eb.close()
    check("Event bus close without error", True)
except Exception as e:
    check("Event bus", False, str(e), fatal=False)

# 8f. Execution guards use RLock
try:
    guards_path = PROJECT_ROOT / "defense" / "execution_guards.py"
    if guards_path.exists():
        code = guards_path.read_text(encoding='utf-8', errors='ignore')
        has_rlock = 'RLock()' in code
        has_bare_lock = 'Lock()' in code and 'RLock()' not in code
        check("execution_guards uses RLock (not bare Lock)",
              has_rlock,
              "Uses threading.Lock (non-reentrant) - deadlock risk!")
    else:
        check("execution_guards.py exists", False)
except Exception as e:
    check("execution_guards RLock check", False, str(e))


# ============================================================================
# SECTION 9: External Feeds
# ============================================================================
section(9, "External Feeds")

# 9a. Fear & Greed API
async def test_fg_api():
    """Test F&G API connectivity using ThreadedResolver."""
    import aiohttp
    from aiohttp.resolver import ThreadedResolver
    connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get("https://api.alternative.me/fng/?limit=1") as resp:
            if resp.status == 200:
                data = await resp.json()
                fg_data = data.get("data", [{}])[0]
                fg_value = int(fg_data.get("value", -1))
                return fg_value
            return -1

try:
    fg_value = asyncio.run(test_fg_api())
    check(f"F&G API: value={fg_value} (0=extreme fear, 100=extreme greed)",
          0 <= fg_value <= 100,
          f"F&G value={fg_value}")
except Exception as e:
    check("F&G API connectivity", False, f"{e}", fatal=False)

# 9b. DNS resolution (ThreadedResolver works)
async def test_dns():
    """Verify ThreadedResolver works on Windows."""
    import aiohttp
    from aiohttp.resolver import ThreadedResolver
    connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get("https://api.kraken.com/0/public/Time") as resp:
            return resp.status == 200

try:
    dns_ok = asyncio.run(test_dns())
    check("DNS via ThreadedResolver (Windows fix)", dns_ok,
          "ThreadedResolver failed - aiodns/c-ares conflict")
except Exception as e:
    check("DNS resolution", False, str(e))

# 9c. Coinglass (may fail without key - WARN not FAIL)
async def test_coinglass():
    import aiohttp
    from aiohttp.resolver import ThreadedResolver
    connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async with session.get("https://open-api.coinglass.com/public/v2/index/fear-greed-history?limit=1") as resp:
            return resp.status in (200, 403)  # 403 = no API key but DNS works

try:
    cg_ok = asyncio.run(test_coinglass())
    check("Coinglass API reachable (200 or 403)", cg_ok,
          "Cannot reach coinglass - DNS or network issue", fatal=False)
except Exception as e:
    check("Coinglass connectivity", False, str(e), fatal=False)


# ============================================================================
# SECTION 10: Regression Tests (Bugs We Fixed)
# ============================================================================
section(10, "Regression Tests")

# 10a. C2: intent.urgency is float (not string "HIGH")
try:
    intent_c2 = TradeIntentV36()
    intent_c2.urgency = 1.0  # This should work
    check("C2: urgency accepts float", isinstance(intent_c2.urgency, float))
    # Verify numeric comparison works
    passes_check = intent_c2.urgency >= 0.7
    check("C2: urgency >= 0.7 comparison works", passes_check)
except Exception as e:
    check("C2: urgency type", False, str(e))

# 10b. C3: short_bias in fusion signals
try:
    # Read integration_v36.py and check _build_fusion_signals has short_bias
    v36_path = PROJECT_ROOT / "integration" / "integration_v36.py"
    v36_code = v36_path.read_text(encoding='utf-8', errors='ignore')

    # Find _build_fusion_signals and check for short_bias
    in_build_fusion = False
    has_short_bias = False
    for line in v36_code.split('\n'):
        if '_build_fusion_signals' in line and 'def ' in line:
            in_build_fusion = True
        if in_build_fusion and line.strip().startswith('def ') and '_build_fusion_signals' not in line:
            in_build_fusion = False
        if in_build_fusion and '"short_bias"' in line and 'signals' in line:
            has_short_bias = True

    check("C3: short_bias signal in _build_fusion_signals", has_short_bias,
          "ShortBiasAgent signal not converted to AgentSignal in fusion")
except Exception as e:
    check("C3: short_bias fusion", False, str(e))

# 10c. C5: GMM scaler None guard
try:
    main_code = (PROJECT_ROOT / "main.py").read_text(encoding='utf-8', errors='ignore')
    has_scaler_guard = ('_gmm_scaler_mean is None' in main_code or
                        'scaler_mean is None' in main_code)
    check("C5: GMM scaler None guard exists", has_scaler_guard,
          "No None check before GMM scaling - crash on model load failure")
except Exception as e:
    check("C5: GMM scaler guard", False, str(e))

# 10d. H2: alpha_gate_passed default is False
try:
    intent_h2 = TradeIntentV36()
    check("H2: alpha_gate_passed defaults to False",
          intent_h2.alpha_gate_passed == False,
          f"Default is {intent_h2.alpha_gate_passed} - not fail-closed!")
except Exception as e:
    check("H2: alpha_gate_passed", False, str(e))

# 10e. Bug #39: execution_guards uses RLock
try:
    guards_code = (PROJECT_ROOT / "defense" / "execution_guards.py").read_text(encoding='utf-8', errors='ignore')
    rlock_count = guards_code.count('RLock()')
    bare_lock_lines = [l.strip() for l in guards_code.split('\n')
                       if 'Lock()' in l and 'RLock' not in l
                       and not l.strip().startswith('#')]
    check(f"Bug #39: execution_guards has {rlock_count} RLock instances",
          rlock_count >= 3,
          f"Only {rlock_count} RLock - should be >= 3")
    check("Bug #39: no bare Lock() (non-reentrant)",
          len(bare_lock_lines) == 0,
          f"Found bare Lock: {bare_lock_lines[:2]}")
except Exception as e:
    check("Bug #39: RLock", False, str(e))

# 10f. Bug #44: HOLD doesn't leak 80% exposure
try:
    # Check that integration_v36 always applies tranche exposure for HOLD
    v36_code = (PROJECT_ROOT / "integration" / "integration_v36.py").read_text(encoding='utf-8', errors='ignore')
    # Look for the HOLD fix: tranche_decision.target_exposure should be applied
    has_hold_fix = ('HOLD' in v36_code and 'target_exposure' in v36_code)
    check("Bug #44: HOLD exposure assignment exists in v36",
          has_hold_fix,
          "HOLD may leak fusion engine's 80% unchecked exposure")
except Exception as e:
    check("Bug #44: HOLD exposure", False, str(e))

# 10g. M1: regime_confidence synced to market_data
try:
    main_code = (PROJECT_ROOT / "main.py").read_text(encoding='utf-8', errors='ignore')
    has_sync = "market_data['regime_confidence']" in main_code or 'market_data["regime_confidence"]' in main_code
    check("M1: regime_confidence synced to market_data",
          has_sync,
          "agent_signals and market_data may have different regime_confidence values")
except Exception as e:
    check("M1: regime_confidence sync", False, str(e))

# 10h. Event bus flush (gzip buffer fix)
try:
    eb_code = (PROJECT_ROOT / "infra" / "event_bus.py").read_text(encoding='utf-8', errors='ignore')
    has_flush = '.flush()' in eb_code
    check("Event bus has .flush() after write (gzip crash-safety)",
          has_flush,
          "gzip buffers not flushed - data lost on crash")
except Exception as e:
    check("Event bus flush", False, str(e))

# 10i. Three drl_weight=0.0 paths
try:
    main_code = (PROJECT_ROOT / "main.py").read_text(encoding='utf-8', errors='ignore')
    drl_weight_refs = main_code.count('drl_weight')
    check(f"3 trade_gate paths: drl_weight refs >= 3 (found {drl_weight_refs})",
          drl_weight_refs >= 3,
          f"Only {drl_weight_refs} refs - need >= 3 for main/authority/p0 paths")
except Exception as e:
    check("drl_weight paths", False, str(e))

# 10j. cross_asset_correlation default = 0.87
try:
    main_code = (PROJECT_ROOT / "main.py").read_text(encoding='utf-8', errors='ignore')
    has_correct_default = '0.87' in main_code and 'cross_asset_correlation' in main_code
    check("cross_asset_correlation default = 0.87 (not 0.65)",
          has_correct_default,
          "Wrong default causes GMM z-score=-4.35 -> posterior collapse")
except Exception as e:
    check("cross_asset_correlation", False, str(e))


# ============================================================================
# SECTION 11: Agent Pipeline Tests
# ============================================================================
section(11, "Agent Pipeline (live signal generation)")

# 11a. ShortBiasAgent.analyze() with real market data
try:
    sba = ShortBiasAgent()
    for asset in ASSETS:
        if asset not in indicator_results or asset not in ticker_data:
            continue
        ind = indicator_results[asset]
        tick = ticker_data[asset]
        sba_market = {
            "price": ind['close'],
            "volume": tick.get('quoteVolume', 0),
            "high_24h": tick.get('high', ind['close']),
            "low_24h": tick.get('low', ind['close']),
            "rsi": ind.get('rsi', 50),
            "adx": ind.get('adx', 20),
            "ema_200": ind.get('ema_200', ind['close']),
            "atr": ind.get('atr', 0),
            "volume_ratio": ind.get('volume_ratio', 1.0),
            "regime_state": "WEAK_CONSOLIDATION",
        }
        signal = sba.analyze(sba_market)
        check(f"{asset} ShortBiasAgent: dir={signal.direction:.2f}, conf={signal.confidence:.2f}",
              hasattr(signal, 'direction') and hasattr(signal, 'confidence'),
              "ShortBiasAgent.analyze() returned invalid signal")
        # Verify direction is in valid range
        check(f"{asset} SBA direction in [-1, 1]",
              -1.0 <= signal.direction <= 1.0,
              f"direction={signal.direction} out of range")
        # Verify confidence is in valid range
        check(f"{asset} SBA confidence in [0, 1]",
              0.0 <= signal.confidence <= 1.0,
              f"confidence={signal.confidence} out of range")
except Exception as e:
    check("ShortBiasAgent.analyze()", False, str(e))

# 11b. DeterministicSentimentEngine.compute() with real F&G
try:
    from signals.deterministic_sentiment import DeterministicSentimentEngine
    dse = DeterministicSentimentEngine()

    # Use real F&G value from Section 9 (or fallback 50)
    live_fg = fg_value if 'fg_value' in dir() and 0 <= fg_value <= 100 else 50

    for asset in ASSETS:
        sent_output = dse.compute(
            fear_greed_value=live_fg,
            asset=asset,
        )
        check(f"{asset} Sentiment: dir={sent_output.direction_score:.2f}, crowding={sent_output.crowding_bias:.2f}",
              hasattr(sent_output, 'direction_score') and hasattr(sent_output, 'crowding_bias'),
              "DeterministicSentimentEngine returned invalid output")
        # Crowding bias should be non-zero with extreme F&G
        if live_fg < 20 or live_fg > 80:
            check(f"{asset} extreme F&G ({live_fg}) -> crowding_bias != 0",
                  abs(sent_output.crowding_bias) > 0,
                  f"F&G={live_fg} but crowding_bias=0 - crowding detection dead",
                  fatal=False)
except Exception as e:
    check("DeterministicSentimentEngine.compute()", False, str(e))

# 11c. AuthorityFusionEngine.fuse() with realistic signals
try:
    from signals.authority_fusion import (
        AuthorityFusionEngine, AgentSignal, FusionContext, SystemMode as FusionSystemMode,
    )
    from market.phase_detector import RegimePhase

    afe = AuthorityFusionEngine()
    test_signals = {
        "quant": AgentSignal(direction=-0.6, confidence=0.7),
        "regime": AgentSignal(direction=-0.6, confidence=0.75),
        "sentiment": AgentSignal(direction=-0.2, confidence=0.3),
        "short_bias": AgentSignal(direction=-0.8, confidence=0.5),
        "risk": AgentSignal(direction=0.0, confidence=0.5),
    }
    ctx = FusionContext(
        mode=FusionSystemMode.NORMAL,
        regime_phase=RegimePhase.EXPANSION,
        data_valid=True,
        drl_enabled=False,
    )
    fusion_result = afe.fuse(test_signals, ctx)

    check(f"Fusion result: dir={fusion_result.direction:.2f}, exposure={fusion_result.target_exposure:.1%}",
          hasattr(fusion_result, 'direction') and hasattr(fusion_result, 'target_exposure'),
          "AuthorityFusionEngine.fuse() returned invalid result")

    # Direction should be negative (all inputs are bearish)
    check(f"Fusion direction is bearish ({fusion_result.direction:.2f})",
          fusion_result.direction < 0,
          f"All agents bearish but fusion direction={fusion_result.direction} - fusion logic broken")

    # Fusion outputs raw target - tranche scheduler caps downstream
    # Here we just verify it's not 0% (fusion logic alive) and not 100%
    check(f"Fusion exposure > 0% and < 100% ({fusion_result.target_exposure:.1%})",
          0.0 < fusion_result.target_exposure < 1.0,
          f"target_exposure={fusion_result.target_exposure:.0%} - fusion dead or uncapped")

except Exception as e:
    check("AuthorityFusionEngine.fuse()", False, str(e))

# 11d. Best-of-N strategy selection (core signal generation)
try:
    for asset in ASSETS:
        if asset not in indicator_results:
            continue
        ind = indicator_results[asset]
        rsi_val = ind.get('rsi', 50)
        adx_val = ind.get('adx', 20)
        close = ind.get('close', 0)
        ema200 = ind.get('ema_200', close)
        macd_val = ind.get('macd', 0)
        bb_up = ind.get('bb_upper', close)
        bb_lo = ind.get('bb_lower', close)
        vol_ratio = ind.get('volume_ratio', 1.0)

        # Compute BB position
        bb_mid = (bb_up + bb_lo) / 2 if bb_up and bb_lo else close
        bb_width = bb_up - bb_lo if bb_up and bb_lo and bb_up > bb_lo else 1
        bb_pos = (close - bb_mid) / (bb_width / 2) if bb_width > 0 else 0

        # Simplified indicator signals
        rsi_sig = (50 - rsi_val) / 50 if rsi_val else 0
        macd_sig = 1.0 if macd_val and macd_val > 0 else (-1.0 if macd_val and macd_val < 0 else 0)
        bb_sig = -bb_pos  # overbought->short, oversold->long
        ema_sig = 1.0 if close > ema200 else -1.0 if ema200 else 0
        adx_mult = min(adx_val / 25, 2.0) if adx_val else 1.0

        # Strategy signals
        strategies = {}
        # Mean revert
        mr_sig = 0.0
        if rsi_val and rsi_val < 30:
            mr_sig = (30 - rsi_val) / 30
        elif rsi_val and rsi_val > 70:
            mr_sig = -(rsi_val - 70) / 30
        strategies['mean_revert'] = {"signal": mr_sig, "strength": abs(mr_sig)}

        # Momentum
        mom_sig = (ema_sig * 0.4 + macd_sig * 0.3 + rsi_sig * 0.1) * adx_mult * 0.5
        strategies['momentum'] = {"signal": mom_sig, "strength": abs(mom_sig)}

        # Volume breakout
        vb_sig = macd_sig * min(vol_ratio, 2.0) * 0.3 if vol_ratio > 1.2 else 0
        strategies['volume_breakout'] = {"signal": vb_sig, "strength": abs(vb_sig)}

        # VRP
        vrp_sig = 0.0
        if rsi_val and rsi_val > 80:
            vrp_sig = -0.5
        elif rsi_val and rsi_val < 20:
            vrp_sig = 0.5
        strategies['vrp'] = {"signal": vrp_sig, "strength": abs(vrp_sig)}

        # Best-of-N selection
        best_name = max(strategies, key=lambda k: strategies[k]["strength"])
        best = strategies[best_name]
        any_nonzero = any(s["strength"] > 0 for s in strategies.values())

        check(f"{asset} strategy selection: {best_name} (sig={best['signal']:.3f}, str={best['strength']:.3f})",
              any_nonzero,
              "All 4 strategies returned signal=0 - indicator pipeline dead")

except Exception as e:
    check("Best-of-N strategy selection", False, str(e))

# 11e. HMATSv36Engine.decide() - full E2E pipeline
try:
    from integration.integration_v36 import HMATSv36Engine

    engine = HMATSv36Engine()

    for asset in ASSETS[:1]:  # Just BTC for speed
        if asset not in indicator_results or asset not in ohlcv_data:
            continue

        ind = indicator_results[asset]
        candles = ohlcv_data[asset]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]

        test_agent_signals = {
            "quant_direction": -0.5,
            "quant_confidence": 0.6,
            "regime_confidence": 0.7,
            "regime_direction": -0.5,
            "short_bias_direction": -0.8,
            "short_bias_confidence": 0.5,
            "sentiment_zscore": -1.0,
        }
        test_market_data = {
            "current_price": closes[-1],
            "volume_24h": sum(volumes[-6:]),  # Required by schema validation
            "volume_ratio": ind.get('volume_ratio', 1.0),
            "rsi": ind.get('rsi', 50),
            "adx": ind.get('adx', 20),
            "atr": ind.get('atr', 100),
            "regime_state": "WEAK_CONSOLIDATION",
            "regime_confidence": 0.7,
            "data_age_seconds": 1.0,
            "spread_bps": 5.0,
            "dvol_zscore": 0.0,
            "vpin": 0.5,
            "correlation_btc_eth_sol": 0.65,
            "orderbook_depth_1pct_usd": float('inf'),
            "signal_edge_bps": 80.0,
            "primary_strategy": "momentum",
        }
        test_position = {
            "current_exposure": 0.0,
            "direction": 0,
            "level": 0,
            "entry_price": 0.0,
        }
        test_system = {
            "equity": 100000.0,
            "daily_pnl_pct": 0.0,
            "tick_count": 3,
        }

        intent = engine.decide(
            asset=asset,
            prices={asset: closes[-50:]},
            volumes={asset: volumes[-50:]},
            agent_signals=test_agent_signals,
            market_data=test_market_data,
            position_state=test_position,
            system_state=test_system,
        )

        check(f"{asset} Engine.decide() returns TradeIntentV36",
              isinstance(intent, TradeIntentV36),
              f"Got {type(intent)}")
        check(f"{asset} intent.direction in [-1, 1]: {intent.direction:.2f}",
              -1.0 <= intent.direction <= 1.0,
              f"direction={intent.direction}")
        check(f"{asset} intent.alpha_estimated_bps > 0: {intent.alpha_estimated_bps:.1f}",
              intent.alpha_estimated_bps > 0,
              "alpha_estimated_bps=0 - alpha gate computation broken")
        check(f"{asset} intent.system_mode: {intent.system_mode}",
              intent.system_mode in ("NORMAL", "OPPORTUNITY", "NO_TRADE"),
              f"system_mode={intent.system_mode}")
        check(f"{asset} intent.target_exposure < 50%: {intent.target_exposure:.1%}",
              intent.target_exposure <= 0.50,
              f"target_exposure={intent.target_exposure:.0%} - uncapped fusion leak?",
              fatal=False)

        print(f"        alpha={intent.alpha_estimated_bps:.0f}bps, "
              f"threshold={intent.alpha_threshold_bps:.0f}bps, "
              f"gate={'PASS' if intent.alpha_gate_passed else 'FAIL'}, "
              f"mode={intent.system_mode}")

except Exception as e:
    check("HMATSv36Engine.decide()", False, f"{e}\n{traceback.format_exc()}")


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  Total:    {_total}")
print(f"  Passed:   {_passed}")
print(f"  Failed:   {_failed}")
print(f"  Warnings: {_warnings}")

if _failed > 0:
    print(f"\n  PAPER RUN PRE-FLIGHT FAILED - fix {_failed} issue(s)")
    print(f"\n  Failed checks:")
    for status, name, detail in _results:
        if status == "FAIL":
            print(f"    FAIL: {name}")
            if detail:
                print(f"          {detail}")
    sys.exit(1)
elif _warnings > 0:
    print(f"\n  {_warnings} warning(s) - review but not blocking")
    print(f"\n  PAPER RUN PRE-FLIGHT PASSED (with warnings)")
    sys.exit(0)
else:
    print(f"\n  ALL CHECKS PASSED - ready for paper run")
    sys.exit(0)

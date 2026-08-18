"""
E2E Full Debug - Tests ALL agents + signal modules (pre-existing + new).
Run: python -X utf8 scripts/e2e_full_debug.py
"""

import sys
import os
import asyncio
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HMATS_INITIAL_CAPITAL", "10000")

errors = []
warnings = []
pass_count = 0


def check(label, condition, detail=""):
    global pass_count
    if condition:
        print(f"  OK: {label}")
        pass_count += 1
    else:
        msg = f"  FAIL: {label} {detail}"
        print(msg)
        errors.append(msg)


def warn(label, detail=""):
    msg = f"  WARN: {label} {detail}"
    print(msg)
    warnings.append(msg)


# Realistic market_data template
MARKET_DATA_BTC = {
    "price": 84500.0,
    "close": 84500.0,
    "open": 85200.0,
    "high": 86000.0,
    "low": 83800.0,
    "volume": 1500000000.0,
    "volume_ratio": 1.3,
    "volume_24h_usd": 36000000000.0,
    "funding_rate": -0.0003,
    "open_interest": 25000000000.0,
    "oi_change_1h_pct": -0.02,
    "oi_change_24h_pct": -0.05,
    "long_liquidations_24h": 150000000.0,
    "short_liquidations_24h": 80000000.0,
    "total_liquidations_24h": 230000000.0,
    "liquidation_imbalance": 0.3,
    "price_change_4h_pct": -0.01,
    "price_change_1h_pct": -0.005,
    "taker_buy_volume": 700000000.0,
    "taker_sell_volume": 800000000.0,
    "atr_14": 1200.0,
    "rsi_14": 42.0,
    "bb_width_20": 0.06,
    "macd_12_26": -150.0,
    "regime_state": "VOLATILE_CHOP",
    "regime_confidence": 0.72,
    "regime_direction": -0.3,
    "spread_bps": 5.0,
    "orderbook_depth_1pct_usd": 5000000.0,
    "order_book_imbalance": -0.15,
    "orderbook_imbalance": -0.15,
    "dvol_zscore": 0.8,
    "vpin": 0.55,
    "correlation_btc_eth_sol": 0.85,
    "sentiment_zscore": -0.3,
    "fear_greed_index": 35,
}


def main():
    print("=" * 70)
    print("  E2E FULL DEBUG - ALL Agents + Signal Modules")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: PRE-EXISTING AGENTS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SECTION 1: PRE-EXISTING AGENTS")
    print("=" * 70)

    # --- ShortBiasAgent ---
    print("\n[AGENT] ShortBiasAgent...")
    try:
        from agents.short_bias_agent import ShortBiasAgent, ShortBiasConfig, ShortSignalType
        sba = ShortBiasAgent(config=ShortBiasConfig())
        check("ShortBias: instantiates", sba is not None)
        check("ShortBias: has generate_signal", hasattr(sba, "generate_signal"))
        check("ShortBias: has _analyze_structure_break (UPG2)",
              hasattr(sba, "_analyze_structure_break"))
        check("ShortBias: STRUCTURE_BREAK in enum",
              hasattr(ShortSignalType, "STRUCTURE_BREAK"))

        # Test generate_signal (no 'now' param - uses asset, price, regime, market_data, derivatives_data, sentiment_data, onchain_data)
        sba_sig = sba.generate_signal(
            asset="BTC",
            price=84500.0,
            regime="VOLATILE_CHOP",
            market_data=MARKET_DATA_BTC,
            derivatives_data={
                "liquidations_long": 150000000,
                "liquidations_short": 80000000,
                "open_interest": 25000000000,
                "funding_rate": -0.0003,
            },
            sentiment_data={"fear_greed_index": 35, "crypto_fear_greed": 35},
            onchain_data={},
        )
        # ShortBiasSignal is a dataclass, not a dict
        check("ShortBias: returns ShortBiasSignal", hasattr(sba_sig, 'direction'))
        check("ShortBias: has confidence attr", hasattr(sba_sig, 'confidence'))
        check("ShortBias: direction in [-1,0]",
              -1.0 <= sba_sig.direction <= 0.0,
              f"got {sba_sig.direction}")
        print(f"  Signal: dir={sba_sig.direction}, conf={sba_sig.confidence}")
    except Exception as e:
        errors.append(f"ShortBiasAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- VolatilityAlphaAgent ---
    print("\n[AGENT] VolatilityAlphaAgent...")
    try:
        from agents.volatility_alpha_agent import VolatilityAlphaAgent, get_volatility_alpha_agent
        va = VolatilityAlphaAgent()
        check("VolAlpha: instantiates", va is not None)
        check("VolAlpha: has generate_signal", hasattr(va, "generate_signal"))
        check("VolAlpha: has to_agent_signal_dict (UPG3)",
              hasattr(va, "to_agent_signal_dict"))

        va_sig = va.generate_signal(
            asset="BTC",
            price=84500.0,
            regime="VOLATILE_CHOP",
        )
        # VolatilityIntent has vol_bias + intensity (NOT direction)
        check("VolAlpha: returns VolatilityIntent", va_sig is not None)
        check("VolAlpha: has vol_bias", hasattr(va_sig, 'vol_bias'))
        check("VolAlpha: has intensity", hasattr(va_sig, 'intensity'))
        print(f"  Signal type: {type(va_sig).__name__}, vol_bias={getattr(va_sig, 'vol_bias', '?')}")
    except Exception as e:
        errors.append(f"VolatilityAlphaAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- MicrostructureArbitrageAgent ---
    print("\n[AGENT] MicrostructureArbitrageAgent...")
    try:
        from agents.microstructure_agent import (
            MicrostructureArbitrageAgent,
            CrossAssetLagDetector,
        )
        ma = MicrostructureArbitrageAgent()
        check("Microstructure: instantiates", ma is not None)
        check("Microstructure: has generate_signal", hasattr(ma, "generate_signal"))
        check("Microstructure: has _cross_asset_lag (UPG5)",
              hasattr(ma, "_cross_asset_lag"))

        # Test CrossAssetLagDetector separately
        cald = CrossAssetLagDetector()
        # Simulate BTC crash, ETH flat
        for p in [85000, 83000, 81000]:
            cald.update_price("BTC", p)
        for p in [3200, 3190, 3185]:
            cald.update_price("ETH", p)
        lag = cald.detect_lag_signal("ETH")
        check("CrossAssetLag: detects BTC->ETH lag", lag is not None)
        if lag:
            check("CrossAssetLag: direction -1", lag["direction"] == -1.0)
        # BTC itself should not have lag
        no_lag = cald.detect_lag_signal("BTC")
        check("CrossAssetLag: BTC self = None", no_lag is None)
    except Exception as e:
        errors.append(f"MicrostructureArbitrageAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- KrakenQuantAgentV6 ---
    print("\n[AGENT] KrakenQuantAgentV6...")
    try:
        from agents.kraken_quant_agent import (
            KrakenQuantAgentV6,
            StrategyAllocator,
            BaseStrategy,
        )
        # UPG1: portfolio_value from env
        sa = StrategyAllocator()
        check("KrakenQuant: StrategyAllocator instantiates", sa is not None)
        check("KrakenQuant: portfolio_value == 10000 (UPG1)",
              sa.portfolio_value == 10000.0,
              f"got {sa.portfolio_value}")

        # UPG6: dynamic stop exists on BaseStrategy
        check("KrakenQuant: BaseStrategy._dynamic_stop exists (UPG6)",
              hasattr(BaseStrategy, "_dynamic_stop"))

        # Test full agent
        kq = KrakenQuantAgentV6()
        check("KrakenQuant: instantiates", kq is not None)
        check("KrakenQuant: has generate_signal", hasattr(kq, "generate_signal"))

        kq_sig = kq.generate_signal(
            asset="BTC",
            market_data=MARKET_DATA_BTC,
            regime="VOLATILE_CHOP",
            regime_confidence=0.72,
        )
        # KrakenQuantPayload is a dataclass, not a dict
        if hasattr(kq_sig, 'to_agent_signal_dict'):
            kq_dict = kq_sig.to_agent_signal_dict()
            check("KrakenQuant: returns KrakenQuantPayload", True)
            check("KrakenQuant: to_agent_signal_dict works", isinstance(kq_dict, dict))
            print(f"  Signal keys: {list(kq_dict.keys())[:10]}")
        elif isinstance(kq_sig, dict):
            check("KrakenQuant: returns dict", True)
            print(f"  Signal keys: {list(kq_sig.keys())[:10]}")
        else:
            check("KrakenQuant: returns known type", False, f"got {type(kq_sig)}")
    except Exception as e:
        errors.append(f"KrakenQuantAgentV6: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- FundingRateAgent ---
    print("\n[AGENT] FundingRateAgent...")
    try:
        from signals.funding_rate_agent import FundingRateAgent, get_funding_rate_agent
        fra = get_funding_rate_agent()
        check("FundingRate: instantiates", fra is not None)
        check("FundingRate: has generate_signal", hasattr(fra, "generate_signal"))

        fr_sig = fra.generate_signal(asset="BTC", funding_rate=-0.0003)
        check("FundingRate: returns something", fr_sig is not None)
        if fr_sig and hasattr(fr_sig, "funding_rate"):
            check("FundingRate: has funding_rate attr", True)
            check("FundingRate: has direction attr", hasattr(fr_sig, "direction"))
        elif isinstance(fr_sig, dict):
            check("FundingRate: returns dict", True)
        else:
            warn("FundingRate: unexpected return type", f"type={type(fr_sig)}")
    except Exception as e:
        errors.append(f"FundingRateAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- DeterministicSentimentEngine ---
    print("\n[AGENT] DeterministicSentimentEngine...")
    try:
        from signals.deterministic_sentiment import (
            DeterministicSentimentEngine,
            get_deterministic_sentiment_engine,
        )
        dse = get_deterministic_sentiment_engine()
        check("DetSentiment: instantiates", dse is not None)
        check("DetSentiment: has compute",
              hasattr(dse, "compute") or hasattr(dse, "compute_sentiment")
              or hasattr(dse, "generate_signal"))
    except Exception as e:
        errors.append(f"DeterministicSentimentEngine: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- ModelAlphaAgent ---
    print("\n[AGENT] ModelAlphaAgent (SHADOW)...")
    try:
        from agents.model_alpha_agent import get_model_alpha_agent
        maa = get_model_alpha_agent()
        check("ModelAlpha: instantiates", maa is not None)
        check("ModelAlpha: has generate_signal", hasattr(maa, "generate_signal"))
    except Exception as e:
        # May fail if dependencies missing - that's OK for shadow mode
        warn(f"ModelAlphaAgent: {e}")

    # --- OnChainGraphAlphaAgent ---
    print("\n[AGENT] OnChainGraphAlphaAgent...")
    try:
        from agents.onchain_graph_alpha import OnChainGraphAlphaAgent
        oga = OnChainGraphAlphaAgent()
        check("OGA: instantiates", oga is not None)
        check("OGA: has generate_signal", hasattr(oga, "generate_signal"))
    except Exception as e:
        errors.append(f"OnChainGraphAlphaAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- OnChainSentimentAlphaEngine ---
    print("\n[AGENT] OnChainSentimentAlphaEngine...")
    try:
        from agents.onchain_sentiment_fusion import (
            OnChainSentimentAlphaEngine,
            get_alpha_engine,
        )
        oce = get_alpha_engine()
        check("OnChainFusion: instantiates", oce is not None)
        check("OnChainFusion: has generate_signals",
              hasattr(oce, "generate_signals") or hasattr(oce, "generate_signal"))
    except Exception as e:
        errors.append(f"OnChainSentimentAlphaEngine: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- SignalEnvelope ---
    print("\n[AGENT] AgentSignalEnvelope...")
    try:
        from agents.signal_envelope import AgentSignalEnvelope
        check("SignalEnvelope: class exists", AgentSignalEnvelope is not None)
        # Check UPG3 extractor
        from agents.signal_envelope import _extract_vol_alpha
        raw = {
            "vol_alpha_implied_direction": -0.3,
            "vol_alpha_intensity": 0.6,
            "vol_alpha_direction_reason": "vol_expand+BEAR->short_confirm",
        }
        extracted = _extract_vol_alpha(raw)
        check("SignalEnvelope: _extract_vol_alpha returns dict",
              isinstance(extracted, dict))
        check("SignalEnvelope: direction == -0.3",
              extracted.get("direction") == -0.3,
              f"got {extracted.get('direction')}")
        check("SignalEnvelope: confidence > 0 when direction != 0",
              extracted.get("confidence", 0) > 0,
              f"got {extracted.get('confidence')}")
    except ImportError:
        warn("SignalEnvelope: _extract_vol_alpha not importable (internal)")
    except Exception as e:
        errors.append(f"AgentSignalEnvelope: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: PRE-EXISTING SIGNAL MODULES
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SECTION 2: PRE-EXISTING SIGNAL MODULES")
    print("=" * 70)

    # --- MacroCrowdAdapter ---
    print("\n[SIGNAL] MacroCrowdAdapter...")
    try:
        from signals.macro_crowd_adapter import (
            MacroCrowdAdapter,
            compute_macro_crowd_effects,
            MacroCrowdEffects,
        )
        check("MacroCrowdAdapter: class exists", MacroCrowdAdapter is not None)
        check("MacroCrowdEffects: class exists", MacroCrowdEffects is not None)

        # Test compute_macro_crowd_effects with all FIX2 scenarios
        scenarios = [
            ("neutral baseline", {
                "macro": {"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": 0.0},
                "crowd": {"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
                "sentiment": {"direction_score": 0.0},
            }),
            ("negative shock (short-favorable)", {
                "macro": {"event_window": {"active": False}, "macro_risk": 0.7, "macro_shock": -0.6},
                "crowd": {"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
                "sentiment": {"direction_score": 0.0},
            }),
            ("positive shock (short-risky)", {
                "macro": {"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.6},
                "crowd": {"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
                "sentiment": {"direction_score": 0.0},
            }),
            ("long crowding (contrarian edge)", {
                "macro": {"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": 0.0},
                "crowd": {"extreme_crowding": True, "crowding_bias": 0.8, "panic_score": 0.0},
                "sentiment": {"direction_score": 0.0},
            }),
            ("short crowding (squeeze risk)", {
                "macro": {"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": 0.0},
                "crowd": {"extreme_crowding": True, "crowding_bias": -0.8, "panic_score": 0.0},
                "sentiment": {"direction_score": 0.0},
            }),
            ("high event importance", {
                "macro": {"event_window": {"active": True, "importance": 4}, "macro_risk": 0.6, "macro_shock": 0.0},
                "crowd": {"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
                "sentiment": {"direction_score": 0.0},
            }),
        ]

        for name, params in scenarios:
            eff = compute_macro_crowd_effects(
                **params, price_direction=-1, system_bias="short",
            )
            check(f"MCE-{name}: returns MacroCrowdEffects",
                  isinstance(eff, MacroCrowdEffects))
            check(f"MCE-{name}: urgency in [0.6,1.0]",
                  0.6 <= eff.urgency_multiplier <= 1.0,
                  f"got {eff.urgency_multiplier:.3f}")
            check(f"MCE-{name}: size in [0.85,1.20]",
                  0.85 <= eff.size_multiplier <= 1.20,
                  f"got {eff.size_multiplier:.3f}")

        # Direction checks
        eff_neg = compute_macro_crowd_effects(
            macro={"event_window": {"active": False}, "macro_risk": 0.7, "macro_shock": -0.6},
            crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
            sentiment={"direction_score": 0.0}, price_direction=-1, system_bias="short",
        )
        eff_pos = compute_macro_crowd_effects(
            macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.6},
            crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
            sentiment={"direction_score": 0.0}, price_direction=-1, system_bias="short",
        )
        check("MCE: neg shock > pos shock (short-bias)",
              eff_neg.size_multiplier > eff_pos.size_multiplier,
              f"neg={eff_neg.size_multiplier:.3f} vs pos={eff_pos.size_multiplier:.3f}")

    except Exception as e:
        errors.append(f"MacroCrowdAdapter: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- ProfitMaxAdapter ---
    print("\n[SIGNAL] ProfitMaxAdapter...")
    try:
        from signals.profit_max_adapter import ProfitMaxAdapter, get_profit_max_adapter
        pma = get_profit_max_adapter()
        check("ProfitMaxAdapter: instantiates", pma is not None)
        check("ProfitMaxAdapter: has evaluate",
              hasattr(pma, "evaluate"))
    except Exception as e:
        errors.append(f"ProfitMaxAdapter: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- FalseBreakoutDetector ---
    print("\n[SIGNAL] FalseBreakoutDetector...")
    try:
        from signals.false_breakout_detector import (
            FalseBreakoutDetector,
            get_false_breakout_detector,
        )
        fbd = get_false_breakout_detector()
        check("FalseBreakout: instantiates", fbd is not None)
        check("FalseBreakout: has detect",
              hasattr(fbd, "detect") or hasattr(fbd, "check_breakout"))
    except Exception as e:
        errors.append(f"FalseBreakoutDetector: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- SignalQualityScorer ---
    print("\n[SIGNAL] SignalQualityScorer...")
    try:
        from signals.signal_quality_scorer import (
            SignalQualityScorer,
            get_signal_quality_scorer,
        )
        sqs = get_signal_quality_scorer()
        check("SignalQuality: instantiates", sqs is not None)
    except Exception as e:
        errors.append(f"SignalQualityScorer: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- FlowSignalIntegrator ---
    print("\n[SIGNAL] FlowSignalIntegrator...")
    try:
        from signals.flow_signal_integrator import (
            FlowSignalIntegrator,
            get_flow_integrator,
        )
        fsi = get_flow_integrator()
        check("FlowIntegrator: instantiates", fsi is not None)
        check("FlowIntegrator: has get_signal_for_symbol",
              hasattr(fsi, "get_signal_for_symbol"))
    except Exception as e:
        errors.append(f"FlowSignalIntegrator: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- AuthorityFusionEngine ---
    print("\n[SIGNAL] AuthorityFusionEngine...")
    try:
        from signals.authority_fusion import (
            AuthorityFusionEngine,
            get_fusion_engine,
        )
        afe = get_fusion_engine()
        check("AuthorityFusion: instantiates", afe is not None)
        check("AuthorityFusion: has fuse",
              hasattr(afe, "fuse") or hasattr(afe, "compute_fusion"))
    except Exception as e:
        errors.append(f"AuthorityFusionEngine: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- RegimeTransitionRisk ---
    print("\n[SIGNAL] RegimeTransitionRisk...")
    try:
        from signals.regime_transition_risk import (
            RegimeTransitionRisk,
            get_regime_transition_risk,
        )
        rtr = get_regime_transition_risk()
        check("RegimeTransitionRisk: instantiates", rtr is not None)
    except Exception as e:
        errors.append(f"RegimeTransitionRisk: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- PartialConsensusChecker ---
    print("\n[SIGNAL] PartialConsensusChecker (SHADOW)...")
    try:
        from signals.partial_consensus import (
            PartialConsensusChecker,
            get_partial_consensus_checker,
        )
        pcc = get_partial_consensus_checker()
        check("PartialConsensus: instantiates", pcc is not None)
    except Exception as e:
        errors.append(f"PartialConsensusChecker: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: NEW MODULES (PHASE 3 + 4)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SECTION 3: NEW MODULES (PHASE 3A/3B/3C + PHASE 4)")
    print("=" * 70)

    # --- 3A: Options Sentiment ---
    print("\n[NEW] OptionsSentimentAgent...")
    try:
        from agents.options_sentiment_agent import (
            get_options_sentiment_agent,
            reset_options_sentiment_agent,
        )
        reset_options_sentiment_agent()
        opt = get_options_sentiment_agent()
        check("3A: singleton works", opt is get_options_sentiment_agent())

        async def _test_3a():
            sig_btc = await opt.generate_signal("BTC", current_price=84500.0)
            check("3A-BTC: has short_confirmation", "short_confirmation" in sig_btc)
            check("3A-BTC: conf in [0,1]", 0 <= sig_btc["confidence"] <= 1)

            sig_eth = await opt.generate_signal("ETH", current_price=3200.0)
            check("3A-ETH: returns dict", isinstance(sig_eth, dict))

            sig_sol = await opt.generate_signal("SOL", current_price=130.0)
            check("3A-SOL: returns neutral", sig_sol["confidence"] == 0.0)

            sig_doge = await opt.generate_signal("DOGE", current_price=0.08)
            check("3A-DOGE: returns neutral", sig_doge["confidence"] == 0.0)
            return sig_btc

        sig_3a = asyncio.run(_test_3a())
    except Exception as e:
        errors.append(f"OptionsSentimentAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- 3B: Squeeze Detector ---
    print("\n[NEW] SqueezeDetectorAgent...")
    try:
        from agents.squeeze_detector_agent import (
            get_squeeze_detector,
            reset_squeeze_detector,
        )
        reset_squeeze_detector()
        sqz = get_squeeze_detector()

        # Normal signal
        sig_n = sqz.generate_signal("BTC", MARKET_DATA_BTC)
        check("3B: normal returns dict", isinstance(sig_n, dict))
        check("3B: squeeze_score in [0,1]", 0 <= sig_n["squeeze_score"] <= 1)

        # Build history then spike
        for i in range(6):
            md = dict(MARKET_DATA_BTC)
            md["short_liquidations_24h"] = 80000000 + i * 5000000
            sqz.generate_signal("BTC", md)

        md_spike = dict(MARKET_DATA_BTC)
        md_spike["short_liquidations_24h"] = 800000000.0
        md_spike["price_change_4h_pct"] = 0.08
        md_spike["oi_change_1h_pct"] = -0.15
        md_spike["funding_rate"] = 0.001
        sig_s = sqz.generate_signal("BTC", md_spike)
        check("3B: spike score > normal", sig_s["squeeze_score"] > sig_n["squeeze_score"])
        check("3B: spike level WARNING or CRITICAL",
              sig_s["squeeze_level"] in ("WARNING", "CRITICAL"),
              f"got {sig_s['squeeze_level']}")

        # Empty market_data - no crash
        sig_e = sqz.generate_signal("ETH", {})
        check("3B: empty market_data -> no crash", isinstance(sig_e, dict))

        # Invalid asset
        sig_inv = sqz.generate_signal("DOGE", MARKET_DATA_BTC)
        check("3B: DOGE -> neutral", sig_inv["squeeze_score"] == 0.0)

        # Check all 6 components exist
        check("3B: has 6 components",
              len(sig_s.get("components", {})) == 6,
              f"got {len(sig_s.get('components', {}))}")

    except Exception as e:
        errors.append(f"SqueezeDetectorAgent: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- 3C: Stablecoin Monitor ---
    print("\n[NEW] StablecoinFlowMonitor...")
    try:
        from signals.stablecoin_monitor import (
            get_stablecoin_monitor,
            reset_stablecoin_monitor,
            StablecoinFlowMonitor,
        )
        reset_stablecoin_monitor()
        stb = get_stablecoin_monitor()

        async def _test_3c():
            sig = await stb.generate_signal()
            check("3C: returns dict", isinstance(sig, dict))
            check("3C: multiplier in [0.7,1.1]",
                  0.7 <= sig["short_size_multiplier"] <= 1.1)
            return sig

        asyncio.run(_test_3c())

        # Multiplier math
        for change, expected_dir in [(-0.02, "above"), (0.02, "below"), (0.0, "equal")]:
            m = StablecoinFlowMonitor._compute_multiplier(change)
            if expected_dir == "above":
                check(f"3C: outflow {change} -> mult > 1", m > 1.0, f"got {m}")
            elif expected_dir == "below":
                check(f"3C: inflow {change} -> mult < 1", m < 1.0, f"got {m}")
            else:
                check(f"3C: neutral -> mult == 1", m == 1.0, f"got {m}")

    except Exception as e:
        errors.append(f"StablecoinFlowMonitor: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- Fringe Panic Detector ---
    print("\n[NEW] FringePanicDetector...")
    try:
        from signals.fringe_panic_detector import (
            get_fringe_panic_detector,
            reset_fringe_panic_detector,
            FringePanicFusion,
        )
        from signals.fringe_panic_keywords import PANIC_KEYWORD_GROUPS

        reset_fringe_panic_detector()
        fpd = get_fringe_panic_detector()

        async def _test_fp():
            await fpd.poll_all()
            sig = fpd.compute_fringe_panic_score(crypto_panic_score=0.4)
            check("FP: returns dict", isinstance(sig, dict))
            check("FP: score in [0,1]", 0 <= sig["fringe_panic_score"] <= 1)
            check("FP: level valid",
                  sig["panic_level"] in ("CALM", "ELEVATED", "HIGH", "EXTREME"))
            check("FP: short_signal valid",
                  sig["short_signal"] in ("NEUTRAL", "FAVORABLE", "SQUEEZE_RISK"))
            return sig

        asyncio.run(_test_fp())

        # Keywords integrity
        check("FP: 8 keyword groups", len(PANIC_KEYWORD_GROUPS) == 8)
        total_kw = sum(len(g["keywords"]) for g in PANIC_KEYWORD_GROUPS.values())
        check("FP: 90+ keywords", total_kw >= 90, f"got {total_kw}")
        wt = sum(g["weight"] for g in PANIC_KEYWORD_GROUPS.values())
        check("FP: weights ~1.0", 0.95 <= wt <= 1.05, f"got {wt:.2f}")

    except Exception as e:
        errors.append(f"FringePanicDetector: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: MACRO FIX1-6 VALIDATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SECTION 4: MACRO FIX1-6 + GlobalContextInformer")
    print("=" * 70)

    # --- GlobalContextInformer ---
    print("\n[MACRO] GlobalContextInformer...")
    try:
        from data_mgmt.global_context_informer import (
            GlobalContextInformer,
            GlobalContextState,
            MacroRegime,
        )
        check("GCI: GlobalContextState exists", GlobalContextState is not None)
        check("GCI: MacroRegime exists", MacroRegime is not None)

        # FIX4: vix_gradient + vix_rising fields
        gcs = GlobalContextState()
        check("FIX4: vix_gradient field exists", hasattr(gcs, "vix_gradient"))
        check("FIX4: vix_rising field exists", hasattr(gcs, "vix_rising"))
        check("FIX4: vix_gradient default = 0.0", gcs.vix_gradient == 0.0)
        check("FIX4: vix_rising default = False", gcs.vix_rising is False)

        # FIX1: MacroRegime enum values
        for regime in ["CRISIS", "DOLLAR_BREAKOUT", "RISK_OFF", "NEUTRAL", "RISK_ON"]:
            check(f"FIX1: MacroRegime.{regime} exists",
                  hasattr(MacroRegime, regime))

    except Exception as e:
        errors.append(f"GlobalContextInformer: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: CROSS-MODULE INTEGRATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SECTION 5: CROSS-MODULE INTEGRATION")
    print("=" * 70)

    # --- Check all agents can produce signals without crashing ---
    print("\n[CROSS] Full agent signal pipeline simulation...")
    try:
        agent_signals = {
            "_signal_timestamp": time.time(),
            "regime_state": "VOLATILE_CHOP",
            "regime_direction": -0.3,
            "regime_confidence": 0.72,
        }

        # ShortBias -> agent_signals (ShortBiasSignal is a dataclass)
        try:
            if sba_sig:
                agent_signals["short_bias_direction"] = sba_sig.direction
                agent_signals["short_bias_confidence"] = sba_sig.confidence
                check("CROSS: short_bias in agent_signals", True)
        except NameError:
            warn("CROSS: sba_sig not available (ShortBias test failed earlier)")

        # Squeeze -> agent_signals
        if 'sig_n' in dir() and sig_n:
            agent_signals["squeeze_score"] = sig_n["squeeze_score"]
            agent_signals["squeeze_level"] = sig_n["squeeze_level"]
            check("CROSS: squeeze in agent_signals", True)

        # FundingRate -> agent_signals
        if 'fr_sig' in dir() and fr_sig:
            if hasattr(fr_sig, "funding_rate"):
                agent_signals["funding_rate"] = fr_sig.funding_rate
            check("CROSS: funding_rate in agent_signals", True)

        # Macro -> agent_signals
        from signals.macro_crowd_adapter import compute_macro_crowd_effects
        mce = compute_macro_crowd_effects(
            macro={"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": -0.3},
            crowd={"extreme_crowding": False, "crowding_bias": 0.1, "panic_score": 0.3},
            sentiment={"direction_score": 0.2},
            price_direction=-1,
            system_bias="short",
        )
        agent_signals["macro_urgency_mult"] = mce.urgency_multiplier
        agent_signals["macro_size_mult"] = mce.size_multiplier
        agent_signals["macro_allow_addon"] = mce.allow_addon
        check("CROSS: macro effects in agent_signals", True)

        # Verify agent_signals has reasonable size
        check("CROSS: agent_signals has 10+ keys",
              len(agent_signals) >= 10,
              f"got {len(agent_signals)}")
        print(f"  agent_signals keys ({len(agent_signals)}): {list(agent_signals.keys())}")

    except Exception as e:
        errors.append(f"Cross-module integration: {e}")
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # --- Check data feed imports (pre-existing) ---
    print("\n[CROSS] Data feed imports...")
    feeds = [
        ("data_mgmt.feeds._http", "create_session"),
        ("data_mgmt.feeds.coinglass_feed", "CoinglassCrowdData"),
        ("data_mgmt.feeds.kraken_futures_feed", "KrakenFuturesFeed"),
    ]
    for mod_name, attr_name in feeds:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            getattr(mod, attr_name)
            check(f"Feed: {mod_name}.{attr_name}", True)
        except Exception as e:
            warn(f"Feed: {mod_name}.{attr_name} -- {e}")

    # --- Check risk modules ---
    print("\n[CROSS] Risk module imports...")
    risk_modules = [
        ("risk.cascade_exhaustion_governor", "get_cascade_exhaustion_governor"),
        ("risk.opportunity_budget_governor", "OpportunityBudgetGovernor"),
    ]
    for mod_name, attr_name in risk_modules:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            getattr(mod, attr_name)
            check(f"Risk: {mod_name}.{attr_name}", True)
        except Exception as e:
            warn(f"Risk: {mod_name}.{attr_name} -- {e}")

    # ═══════════════��═══════════════════════════════════════════
    # SECTION 6: VERIFY MODE SMOKE TEST
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SECTION 6: INVARIANT CHECKS")
    print("=" * 70)

    # No DRL obs_dim contamination
    import subprocess
    result = subprocess.run(
        ["python", "-c",
         "import grep_check; print('skip')"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
    encoding="utf-8")
    # Manual check instead
    print("\n[INV] DRL safety invariants...")
    new_files = [
        "agents/options_sentiment_agent.py",
        "agents/squeeze_detector_agent.py",
        "signals/stablecoin_monitor.py",
        "signals/fringe_panic_detector.py",
        "signals/fringe_panic_keywords.py",
    ]
    root = os.path.dirname(os.path.dirname(__file__))
    for f in new_files:
        fpath = os.path.join(root, f)
        if os.path.exists(fpath):
            content = open(fpath, encoding="utf-8").read()
            check(f"INV: {f} no obs_dim", "obs_dim" not in content)
            check(f"INV: {f} no feature_manifest", "feature_manifest" not in content)
        else:
            warn(f"INV: {f} not found")

    # No veto in new modules
    for f in new_files:
        fpath = os.path.join(root, f)
        if os.path.exists(fpath):
            content = open(fpath, encoding="utf-8").read()
            # Check for veto_active=True (actual veto implementation)
            has_veto_impl = "veto_active=True" in content or "veto_active = True" in content
            check(f"INV: {f} no veto impl", not has_veto_impl)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    if errors:
        print(f"  RESULT: {len(errors)} FAILURES, {len(warnings)} warnings, {pass_count} passed")
        for e in errors:
            print(f"    {e}")
    else:
        print(f"  RESULT: ALL {pass_count} TESTS PASSED ({len(warnings)} warnings)")

    if warnings:
        print(f"\n  Warnings (non-critical):")
        for w in warnings:
            print(f"    {w}")

    print("=" * 70)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
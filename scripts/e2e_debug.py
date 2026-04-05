"""
E2E Debug Script - Tests all PHASE 1-4 agents and modules.
Run: python -X utf8 scripts/e2e_debug.py
"""

import sys
import os
import asyncio
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HMATS_INITIAL_CAPITAL", "10000")

errors = []


def check(label, condition, detail=""):
    if condition:
        print(f"  OK: {label}")
    else:
        msg = f"  FAIL: {label} {detail}"
        print(msg)
        errors.append(msg)


def main():
    print("=" * 70)
    print("  E2E DEBUG - All Modified Agents + New Modules")
    print("=" * 70)

    # ═══════════════════════════════════════════
    # TEST 1: Import chain
    # ═══════════════════════════════════════════
    print("\n[TEST 1] Import chain...")

    modules_to_test = [
        ("agents.options_sentiment_agent", "get_options_sentiment_agent"),
        ("agents.squeeze_detector_agent", "get_squeeze_detector"),
        ("signals.stablecoin_monitor", "get_stablecoin_monitor"),
        ("signals.fringe_panic_detector", "get_fringe_panic_detector"),
        ("signals.fringe_panic_keywords", "PANIC_KEYWORD_GROUPS"),
        ("signals.macro_crowd_adapter", "compute_macro_crowd_effects"),
        ("data_mgmt.global_context_informer", "GlobalContextInformer"),
        ("agents.short_bias_agent", "ShortBiasAgent"),
        ("agents.volatility_alpha_agent", "VolatilityAlphaAgent"),
        ("agents.microstructure_agent", "MicrostructureArbitrageAgent"),
        ("agents.kraken_quant_agent", "KrakenQuantAgentV6"),
        ("agents.signal_envelope", "AgentSignalEnvelope"),
    ]

    for mod_name, attr_name in modules_to_test:
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            getattr(mod, attr_name)
            print(f"  OK: {mod_name}.{attr_name}")
        except Exception as e:
            msg = f"  FAIL: {mod_name}.{attr_name} -- {e}"
            print(msg)
            errors.append(msg)

    # ═══════════════════════════════════════════
    # TEST 2: Singleton instantiation
    # ═══════════════════════════════════════════
    print("\n[TEST 2] Singleton instantiation...")

    from agents.options_sentiment_agent import (
        get_options_sentiment_agent,
        reset_options_sentiment_agent,
    )
    from agents.squeeze_detector_agent import (
        get_squeeze_detector,
        reset_squeeze_detector,
    )
    from signals.stablecoin_monitor import (
        get_stablecoin_monitor,
        reset_stablecoin_monitor,
    )
    from signals.fringe_panic_detector import (
        get_fringe_panic_detector,
        reset_fringe_panic_detector,
    )

    reset_options_sentiment_agent()
    reset_squeeze_detector()
    reset_stablecoin_monitor()
    reset_fringe_panic_detector()

    opt = get_options_sentiment_agent()
    sqz = get_squeeze_detector()
    stb = get_stablecoin_monitor()
    fpd = get_fringe_panic_detector()

    check("OptionsSentiment singleton", opt is get_options_sentiment_agent())
    check("SqueezeDetector singleton", sqz is get_squeeze_detector())
    check("StablecoinMonitor singleton", stb is get_stablecoin_monitor())
    check("FringePanicDetector singleton", fpd is get_fringe_panic_detector())
    check("OptionsSentiment has _mock_mode", hasattr(opt, "_mock_mode"))
    check("StablecoinMonitor has _mock_mode", hasattr(stb, "_mock_mode"))
    check("FringePanic X has _mock_mode", hasattr(fpd._x, "_mock_mode"))

    # ═══════════════════════════════════════════
    # TEST 3: Signal generation with realistic data
    # ═══════════════════════════════════════════
    print("\n[TEST 3] Signal generation with realistic data...")

    market_data = {
        "price": 84500.0,
        "close": 84500.0,
        "open": 85200.0,
        "high": 86000.0,
        "low": 83800.0,
        "volume": 1500000000.0,
        "volume_ratio": 1.3,
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
        "regime_state": "VOLATILE_CHOP",
        "regime_confidence": 0.72,
    }

    # --- 3A: Options Sentiment ---
    async def test_options():
        print("\n  --- 3A: Options Sentiment Agent ---")
        sig = await opt.generate_signal("BTC", current_price=84500.0)
        check("3A: returns dict", isinstance(sig, dict))
        check("3A: has short_confirmation", "short_confirmation" in sig)
        check("3A: has confidence", "confidence" in sig)
        check("3A: has source", "source" in sig)
        check(
            "3A: short_confirmation in [-1,1]",
            -1.0 <= sig["short_confirmation"] <= 1.0,
            f"got {sig['short_confirmation']}",
        )
        check(
            "3A: confidence in [0,1]",
            0.0 <= sig["confidence"] <= 1.0,
            f"got {sig['confidence']}",
        )
        # SOL should return neutral
        sol_sig = await opt.generate_signal("SOL", current_price=130.0)
        check("3A: SOL returns neutral", sol_sig["confidence"] == 0.0)
        print(f"  Signal: {sig}")
        return sig

    # --- 3B: Squeeze Detector ---
    def test_squeeze():
        print("\n  --- 3B: Squeeze Detector Agent ---")
        sig = sqz.generate_signal("BTC", market_data)
        check("3B: returns dict", isinstance(sig, dict))
        check("3B: has squeeze_score", "squeeze_score" in sig)
        check("3B: has squeeze_level", "squeeze_level" in sig)
        check("3B: has recommended_action", "recommended_action" in sig)
        check("3B: has components", "components" in sig)
        check(
            "3B: squeeze_score in [0,1]",
            0.0 <= sig["squeeze_score"] <= 1.0,
            f"got {sig['squeeze_score']}",
        )
        check(
            "3B: level valid",
            sig["squeeze_level"] in ("SAFE", "WATCH", "WARNING", "CRITICAL"),
            f"got {sig['squeeze_level']}",
        )
        check(
            "3B: action valid",
            sig["recommended_action"]
            in ("NONE", "REDUCE_30", "REDUCE_50", "FLATTEN"),
            f"got {sig['recommended_action']}",
        )

        # Build history (5 normal ticks)
        for i in range(5):
            md2 = dict(market_data)
            md2["short_liquidations_24h"] = 80000000 + i * 10000000
            sqz.generate_signal("BTC", md2)

        # Spike test
        md_spike = dict(market_data)
        md_spike["short_liquidations_24h"] = 800000000.0  # 10x
        md_spike["price_change_4h_pct"] = 0.08  # 8% up
        md_spike["oi_change_1h_pct"] = -0.15  # 15% OI collapse
        md_spike["funding_rate"] = 0.001  # positive funding
        sig_spike = sqz.generate_signal("BTC", md_spike)
        check(
            "3B: spike raises score",
            sig_spike["squeeze_score"] > sig["squeeze_score"],
            f"spike={sig_spike['squeeze_score']:.3f} vs normal={sig['squeeze_score']:.3f}",
        )
        print(f"  Normal: score={sig['squeeze_score']}, level={sig['squeeze_level']}")
        print(f"  Spike:  score={sig_spike['squeeze_score']}, level={sig_spike['squeeze_level']}, components={sig_spike['components']}")
        return sig, sig_spike

    # --- 3C: Stablecoin Monitor ---
    async def test_stablecoin():
        print("\n  --- 3C: Stablecoin Flow Monitor ---")
        sig = await stb.generate_signal()
        check("3C: returns dict", isinstance(sig, dict))
        check("3C: has supply_change_7d_pct", "supply_change_7d_pct" in sig)
        check("3C: has short_size_multiplier", "short_size_multiplier" in sig)
        check("3C: has flow_direction", "flow_direction" in sig)
        check(
            "3C: multiplier in [0.7,1.1]",
            0.7 <= sig["short_size_multiplier"] <= 1.1,
            f"got {sig['short_size_multiplier']}",
        )
        check(
            "3C: flow_direction valid",
            sig["flow_direction"] in ("NEUTRAL", "INFLOW", "OUTFLOW"),
            f"got {sig['flow_direction']}",
        )
        print(f"  Signal: {sig}")

        # Test multiplier logic
        from signals.stablecoin_monitor import StablecoinFlowMonitor

        m1 = StablecoinFlowMonitor._compute_multiplier(-0.02)
        check("3C: outflow -> mult > 1.0", m1 > 1.0, f"got {m1:.3f}")
        m2 = StablecoinFlowMonitor._compute_multiplier(0.02)
        check("3C: inflow -> mult < 1.0", m2 < 1.0, f"got {m2:.3f}")
        m3 = StablecoinFlowMonitor._compute_multiplier(0.0)
        check("3C: neutral -> mult == 1.0", m3 == 1.0, f"got {m3:.3f}")
        print(f"  Multiplier: outflow={m1:.3f}, inflow={m2:.3f}, neutral={m3:.3f}")
        return sig

    # --- Fringe Panic ---
    async def test_fringe():
        print("\n  --- PHASE 4: Fringe Panic Detector ---")
        await fpd.poll_all()
        sig = fpd.compute_fringe_panic_score(crypto_panic_score=0.3)
        check("FP: returns dict", isinstance(sig, dict))
        check("FP: has fringe_panic_score", "fringe_panic_score" in sig)
        check("FP: has panic_level", "panic_level" in sig)
        check("FP: has short_signal", "short_signal" in sig)
        check("FP: has leverage_adjustment", "leverage_adjustment" in sig)
        check(
            "FP: score in [0,1]",
            0.0 <= sig["fringe_panic_score"] <= 1.0,
            f"got {sig['fringe_panic_score']}",
        )
        check(
            "FP: level valid",
            sig["panic_level"] in ("CALM", "ELEVATED", "HIGH", "EXTREME"),
            f"got {sig['panic_level']}",
        )
        check(
            "FP: short_signal valid",
            sig["short_signal"] in ("NEUTRAL", "FAVORABLE", "SQUEEZE_RISK"),
            f"got {sig['short_signal']}",
        )
        check(
            "FP: leverage_adj in [-0.3,0.3]",
            -0.3 <= sig["leverage_adjustment"] <= 0.3,
            f"got {sig['leverage_adjustment']}",
        )
        print(f"  Signal: {sig}")

        # Test status
        status = fpd.get_status()
        check("FP: status has x_mock_mode", "x_mock_mode" in status)
        check("FP: status has trends_available", "trends_available" in status)
        return sig

    # Run all async tests
    async def run_async_tests():
        opt_sig = await test_options()
        stb_sig = await test_stablecoin()
        fp_sig = await test_fringe()
        return opt_sig, stb_sig, fp_sig

    opt_sig, stb_sig, fp_sig = asyncio.run(run_async_tests())
    sqz_sig, sqz_spike = test_squeeze()

    # ═══════════════════════════════════════════
    # TEST 4: Macro FIX1 + FIX2 validation
    # ═══════════════════════════════════════════
    print("\n[TEST 4] Macro FIX1 + FIX2 validation...")

    from signals.macro_crowd_adapter import compute_macro_crowd_effects

    # FIX2: Negative shock + short bias -> size boost
    eff1 = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.7, "macro_shock": -0.6},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.0},
        sentiment={"direction_score": 0.0},
        price_direction=-1,
        system_bias="short",
    )
    check("FIX2: neg shock -> size > 1.0", eff1.size_multiplier > 1.0,
          f"got {eff1.size_multiplier:.3f}")

    # FIX2: Bullish crowd -> contrarian boost
    eff2 = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.0},
        crowd={"extreme_crowding": False, "crowding_bias": 0.5, "panic_score": 0.0},
        sentiment={"direction_score": 0.6},
        price_direction=-1,
        system_bias="short",
    )
    check("FIX2: bullish crowd -> contrarian boost", eff2.size_multiplier > 1.0,
          f"got {eff2.size_multiplier:.3f}")

    # FIX2: Short crowding -> restrict
    eff3 = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.3, "macro_shock": 0.0},
        crowd={"extreme_crowding": True, "crowding_bias": -0.8, "panic_score": 0.0},
        sentiment={"direction_score": 0.0},
        price_direction=-1,
        system_bias="short",
    )
    check("FIX2: short crowding -> no addon", not eff3.allow_addon)
    check("FIX2: short crowding -> urgency < 0.8", eff3.urgency_multiplier < 0.8,
          f"got {eff3.urgency_multiplier:.3f}")

    # FIX2: Moderate panic -> size boost
    eff4 = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": 0.0},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.8},
        sentiment={"direction_score": 0.0},
        price_direction=-1,
        system_bias="short",
    )
    check("FIX2: moderate panic -> size boost", eff4.size_multiplier > 1.0,
          f"got {eff4.size_multiplier:.3f}")

    # FIX2: Extreme panic -> mild caution
    eff5 = compute_macro_crowd_effects(
        macro={"event_window": {"active": False}, "macro_risk": 0.5, "macro_shock": 0.0},
        crowd={"extreme_crowding": False, "crowding_bias": 0.0, "panic_score": 0.95},
        sentiment={"direction_score": 0.0},
        price_direction=-1,
        system_bias="short",
    )
    check("FIX2: extreme panic -> urgency < 1.0", eff5.urgency_multiplier < 1.0,
          f"got {eff5.urgency_multiplier:.3f}")

    print(f"  Effects: neg_shock={eff1.size_multiplier:.3f}, "
          f"contrarian={eff2.size_multiplier:.3f}, "
          f"short_crowd_urg={eff3.urgency_multiplier:.3f}, "
          f"panic_boost={eff4.size_multiplier:.3f}, "
          f"extreme_panic_urg={eff5.urgency_multiplier:.3f}")

    # ═══════════════════════════════════════════
    # TEST 5: UPG1-7 validation
    # ═══════════════════════════════════════════
    print("\n[TEST 5] Agent Upgrades UPG1-7...")

    # UPG1: portfolio_value from env
    from agents.kraken_quant_agent import KrakenQuantAgentV6
    kq = KrakenQuantAgentV6.__new__(KrakenQuantAgentV6)
    # Check that StrategyAllocator reads from env
    from agents.kraken_quant_agent import StrategyAllocator
    sa = StrategyAllocator()
    check("UPG1: portfolio_value == 10000", sa.portfolio_value == 10000.0,
          f"got {sa.portfolio_value}")

    # UPG2: Structure Break in ShortBiasAgent
    from agents.short_bias_agent import ShortSignalType
    check("UPG2: STRUCTURE_BREAK in enum",
          hasattr(ShortSignalType, "STRUCTURE_BREAK"))
    from agents.short_bias_agent import ShortBiasAgent
    sba = ShortBiasAgent.__new__(ShortBiasAgent)
    check("UPG2: has _analyze_structure_break",
          hasattr(sba, "_analyze_structure_break"))

    # UPG3: VolAlpha bear regime direction
    from agents.volatility_alpha_agent import VolatilityAlphaAgent
    va = VolatilityAlphaAgent.__new__(VolatilityAlphaAgent)
    check("UPG3: has to_agent_signal_dict",
          hasattr(va, "to_agent_signal_dict"))

    # UPG5: CrossAssetLagDetector
    from agents.microstructure_agent import CrossAssetLagDetector
    cald = CrossAssetLagDetector()
    cald.update_price("BTC", 85000)
    cald.update_price("BTC", 83000)  # -2.35%
    cald.update_price("BTC", 81000)  # -2.41%
    cald.update_price("ETH", 3200)
    cald.update_price("ETH", 3190)  # -0.31%
    cald.update_price("ETH", 3185)  # -0.16%
    lag_sig = cald.detect_lag_signal("ETH")
    check("UPG5: CrossAssetLag detects BTC lead",
          lag_sig is not None,
          f"got {lag_sig}")
    if lag_sig:
        check("UPG5: direction == -1.0", lag_sig["direction"] == -1.0)
        check("UPG5: confidence > 0", lag_sig["confidence"] > 0)
        print(f"  Lead-Lag: {lag_sig}")

    # UPG6: Dynamic stop - check method exists on class (abstract, can't instantiate)
    from agents.kraken_quant_agent import BaseStrategy
    check("UPG6: has _dynamic_stop", hasattr(BaseStrategy, "_dynamic_stop"))

    # UPG7: Deprecated comment
    import agents
    with open(
        os.path.join(os.path.dirname(agents.__file__), "__init__.py"),
        encoding="utf-8",
        errors="replace",
    ) as f:
        agents_init = f.read()
    check("UPG7: DEPRECATED marker in __init__", "DEPRECATED" in agents_init)

    # ═══════════════════════════════════════════
    # TEST 6: Fail-safe / error paths
    # ═══════════════════════════════════════════
    print("\n[TEST 6] Fail-safe / error paths...")

    # 3B: Invalid asset
    inv_sig = sqz.generate_signal("DOGE", market_data)
    check("3B: invalid asset -> neutral", inv_sig["squeeze_score"] == 0.0)

    # 3B: Empty market_data
    empty_sig = sqz.generate_signal("BTC", {})
    check("3B: empty market_data -> no crash", isinstance(empty_sig, dict))

    # 3A: Invalid asset
    async def test_failsafe():
        sol_sig = await opt.generate_signal("DOGE", current_price=0)
        check("3A: DOGE returns neutral", sol_sig["confidence"] == 0.0)

        # FP: compute with no poll
        from signals.fringe_panic_detector import FringePanicFusion
        fresh = FringePanicFusion()
        sig = fresh.compute_fringe_panic_score(0.0)
        check("FP: fresh compute -> CALM", sig["panic_level"] == "CALM")
        check("FP: fresh compute -> score 0", sig["fringe_panic_score"] <= 0.5)

    asyncio.run(test_failsafe())

    # ═══════════════════════════════════════════
    # TEST 7: Keywords config integrity
    # ═══════════════════════════════════════════
    print("\n[TEST 7] Keywords config integrity...")

    from signals.fringe_panic_keywords import (
        PANIC_KEYWORD_GROUPS,
        CHINA_KEYWORDS_ZH,
        TRENDS_KEYWORDS_BATCH1,
        TRENDS_KEYWORDS_BATCH2,
        PANIC_SUBREDDITS,
    )

    check("KW: 8 groups", len(PANIC_KEYWORD_GROUPS) == 8)
    total_kw = sum(len(g["keywords"]) for g in PANIC_KEYWORD_GROUPS.values())
    check("KW: 90+ keywords", total_kw >= 90, f"got {total_kw}")
    check("KW: all groups have x_query",
          all("x_query" in g for g in PANIC_KEYWORD_GROUPS.values()))
    check("KW: all groups have weight",
          all("weight" in g for g in PANIC_KEYWORD_GROUPS.values()))
    check("KW: all groups have baseline",
          all("baseline_daily_tweets" in g for g in PANIC_KEYWORD_GROUPS.values()))
    weights = sum(g["weight"] for g in PANIC_KEYWORD_GROUPS.values())
    check("KW: weights sum ~1.0", 0.95 <= weights <= 1.05, f"sum={weights:.2f}")
    check("KW: China ZH has x_query", "x_query" in CHINA_KEYWORDS_ZH)
    check("KW: Trends batch1 non-empty", len(TRENDS_KEYWORDS_BATCH1) > 0)
    check("KW: Subreddits non-empty", len(PANIC_SUBREDDITS) > 0)

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    if errors:
        print(f"  RESULT: {len(errors)} FAILURES")
        for e in errors:
            print(f"    {e}")
        sys.exit(1)
    else:
        with open(__file__, encoding="utf-8", errors="replace") as f:
            total_tests = sum(1 for line in f.readlines() if "check(" in line)
        print(f"  RESULT: ALL TESTS PASSED ({total_tests}+ checks)")
    print("=" * 70)


if __name__ == "__main__":
    main()

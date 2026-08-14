"""[P265] Decision-path wiring fixes.

Six independent reader/writer or ordering defects on the live decision path:

  1. alpha_boost read the engine-GLOBAL `_last_phase_result`, whose writers
     run later in the tick — ETH's transition-aggression boost was gated on
     BTC's phase confidence every tick (the P225 leak, in the consumer P225's
     per-asset fix missed).
  2. The trend layer's enforce-inject wrote its freshness stamp
     (quant_data_quality=1.0) into market_data only; fusion's P170 guard
     reads agent_signals (copied BEFORE the inject) — on pipeline-degraded
     ticks the live driver's signal was silently excluded from fusion.
     (Same gap in the P256 regimebook seat.)
  3. The DRL shadow-diag block read _ood_score/_drl_drift_weight ~900 lines
     BEFORE their only writers, in a dict rebuilt fresh each tick — every
     readiness-ledger record carried ood=0.0/drift=1.0 constants.
  4. main.py fed kraken_quant a fabricated taker_ratio=1.0 (no producer of
     the key exists anywhere) — the absorption buffer compared constants
     against >1.0 (unreachable) and _has_field counted the constant as
     present data.
  5. main.py overwrote the pipeline's rolling ofi_zscore (clipped ±5) with
     the micro agent's RAW [-1,1] imbalance — arming a dead |z|>3 toxicity
     veto miscalibration the day micro's warm-up completes.
  6. Attribution dq was fiction in both directions: quant collected a
     producer-less "data_quality" key (pinned 0.0) while agents with no dq
     key got a fabricated 1.0 from the extractor fallback. Plus: micro's dq
     was clobbered with a default-1.0 read of a key its payload never
     carries; and RegimeICFusion's stable-sign set named "funding" while the
     caller says "funding_rate" (funding was regime-sign-flipped in the
     [RIC-SHADOW] evidence stream against the OOS design).
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN_SRC = (REPO / "main.py").read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 1. alpha_boost phase input is per-asset
# ---------------------------------------------------------------------------

class TestAlphaBoostPhaseInput:
    def _block(self):
        start = MAIN_SRC.index("if self._alpha_boost:")
        end = MAIN_SRC.index("_ab_result = self._alpha_boost.compute", start)
        return MAIN_SRC[start:end]

    def test_reads_the_per_asset_store(self):
        assert "_last_phase_by_asset" in self._block(), (
            "alpha_boost no longer reads the per-asset phase — the engine-"
            "global slot holds the PREVIOUS asset's phase at this point in "
            "the tick (P225/P265 cross-asset leak)")

    def test_does_not_read_the_global_slot(self):
        # Assert the READ shape (a comment naming the slot is fine — the
        # P177 comment-trap; the first version of this test tripped on the
        # fix's own explanatory comment).
        blk = self._block()
        assert "getattr(self.engine" not in blk and \
            "self.engine._last_phase_result" not in blk, (
            "alpha_boost reads the engine-global phase slot again — that "
            "slot is written LATER in the tick, so this gates one asset's "
            "boost on another asset's phase")


# ---------------------------------------------------------------------------
# 2. enforce-inject freshness stamps reach fusion's dict
# ---------------------------------------------------------------------------

class TestTrendInjectFreshnessStamp:
    def test_trend_layer_stamps_both_dicts(self):
        import types
        from core.trend_decision_layer import TrendDecisionLayer
        layer = TrendDecisionLayer(mode="enforce")
        # Stub the strategy: the test targets the INJECT's dict writes, not
        # the signal formula.
        layer._strat = types.SimpleNamespace(
            compute=lambda closes: {"signal": 0.9},
            min_history=lambda: 10)
        layer._closes["BTC"] = [100.0 + i * 0.5 for i in range(300)]
        agent_signals: dict = {"quant_data_quality": 0.0}  # degraded pipeline
        market_data: dict = {"quant_direction": 0.0}
        layer.process("BTC", None, agent_signals, market_data)
        assert agent_signals.get("quant_direction"), "inject did not fire — fixture too weak"
        assert agent_signals.get("quant_data_quality") == 1.0, (
            "the enforce-inject left quant_data_quality=0.0 in agent_signals "
            "— fusion's P170 guard reads THAT dict, so the freshly computed "
            "trend signal is excluded from fusion on every degraded tick")
        assert market_data.get("quant_data_quality") == 1.0

    def test_regimebook_seat_stamps_both_dicts(self):
        start = MAIN_SRC.index("[REGIMEBOOK-SEAT]")
        seg_start = MAIN_SRC.rindex("_rb_mode", 0, start)
        seg = MAIN_SRC[seg_start:seg_start + 4000]
        assert 'agent_signals["quant_data_quality"]' in seg, (
            "the regimebook seat writes direction/confidence into "
            "agent_signals but not the dq stamp — same exclusion gap")


# ---------------------------------------------------------------------------
# 3. DRL shadow diag reads after its writers
# ---------------------------------------------------------------------------

class TestDrlShadowDiagOrdering:
    def test_reader_follows_both_writers(self):
        reader = MAIN_SRC.index('_dsd_ood = agent_signals.get("_ood_score"')
        ood_writer = MAIN_SRC.index('agent_signals["_ood_score"] =')
        drift_writer = MAIN_SRC.index('agent_signals["_drl_drift_weight"] =')
        assert reader > ood_writer and reader > drift_writer, (
            "the [DRL_SHADOW_DIAG] reader precedes its writers again — in a "
            "dict rebuilt fresh each tick that records the constants "
            "ood=0.0/drift=1.0 into the Rung-3 readiness ledger (P234 shape)")


# ---------------------------------------------------------------------------
# 4. no fabricated taker_ratio
# ---------------------------------------------------------------------------

class TestTakerRatio:
    def test_no_fabricated_default_anywhere(self):
        assert 'get("taker_ratio", 1.0)' not in MAIN_SRC, (
            "a fabricated taker_ratio=1.0 default is back — no producer of "
            "that key exists; the constant fills kraken_quant's absorption "
            "buffer and fakes _has_field data presence")

    def test_converter_treats_absence_as_absence(self):
        from agents.kraken_quant_agent import KrakenQuantAgent
        nested = KrakenQuantAgent._convert_market_data(
            "BTC", {"current_price": 100.0})
        assert nested["taker_ratio"] == {}, (
            "an absent taker_ratio key produced entries in the nested dict")

    def test_converter_passes_a_real_ratio_through(self):
        from agents.kraken_quant_agent import KrakenQuantAgent
        nested = KrakenQuantAgent._convert_market_data(
            "BTC", {"taker_ratio_btc": 1.37})
        assert nested["taker_ratio"].get("BTC") == pytest.approx(1.37)

    def test_the_real_ratio_is_derived_from_binance_volumes(self):
        seg_start = MAIN_SRC.index("[BINANCE_LOB]")
        seg = MAIN_SRC[max(0, seg_start - 2500):seg_start + 500]
        assert 'market_data["taker_ratio"]' in seg, (
            "the Binance block no longer derives the real taker ratio — "
            "kraken_quant's absorption dimension is starved again")


# ---------------------------------------------------------------------------
# 5. ofi_zscore is not overwritten with a raw imbalance
# ---------------------------------------------------------------------------

class TestOfiZscoreNotClobbered:
    def test_the_overwrite_is_gone(self):
        assert re.search(
            r"market_data\[.ofi_zscore.\]\s*=\s*_micro_sig", MAIN_SRC) is None, (
            "main.py overwrites the pipeline's rolling ofi_zscore with the "
            "micro agent's raw [-1,1] imbalance again — every z-calibrated "
            "consumer (|z|>3 SOL toxicity veto included) silently receives a "
            "variable that cannot exceed ~1.0")


# ---------------------------------------------------------------------------
# 6. attribution dq honesty
# ---------------------------------------------------------------------------

class TestAttributionDataQuality:
    def test_absence_is_the_sentinel_not_fabricated_health(self):
        from agents.signal_envelope import _extract_data_quality, DQ_NOT_REPORTED
        assert _extract_data_quality({}) == DQ_NOT_REPORTED
        assert _extract_data_quality({"direction": 0.5}) == DQ_NOT_REPORTED

    def test_a_real_zero_is_still_zero(self):
        from agents.signal_envelope import _extract_data_quality
        assert _extract_data_quality({"micro_data_quality": 0.0}) == 0.0, (
            "a measured dq of 0.0 (the degraded-path truth) was not passed "
            "through — 0.0 is a measurement, not absence")

    def test_quant_collects_the_real_dq_key(self):
        start = MAIN_SRC.index('"quant": {**{k: agent_signals.get(k, 0.0)')
        seg = MAIN_SRC[start:start + 1200]
        assert 'agent_signals.get("quant_data_quality"' in seg, (
            "quant's attribution entry no longer collects quant_data_quality "
            "— the DECIDE agent's envelope dq returns to a producer-less key")
        assert '"primary_strategy", "data_quality"' not in seg

    def test_micro_dq_is_not_clobbered_with_a_default(self):
        assert "_micro_sig.get('data_quality'" not in MAIN_SRC, (
            "the micro block reads the payload's nonexistent 'data_quality' "
            "key again — always the default, clobbering the real "
            "micro_data_quality the wholesale copy just wrote")


class TestRegimeICStableSign:
    def test_the_set_names_the_callers_agents(self):
        from signals.regime_ic_fusion import RegimeICFusion
        m = re.search(
            r"for a, k in \((.*?)\)\}", MAIN_SRC, re.S)
        assert m, "could not locate the RIC caller roster in main.py"
        roster = set(re.findall(r'\("([a-z_]+)",\s*"', m.group(1)))
        assert roster, "roster parse came back empty"
        missing = RegimeICFusion.DEFAULT_STABLE_SIGN - roster
        assert not missing, (
            f"stable-sign members {missing} match NO agent the caller feeds "
            f"(roster={sorted(roster)}) — those agents are silently "
            f"regime-sign-flipped in the [RIC-SHADOW] evidence stream, the "
            f"exact 'funding' vs 'funding_rate' mismatch of P265")

    def test_funding_rate_is_stable_sign(self):
        from signals.regime_ic_fusion import RegimeICFusion
        assert "funding_rate" in RegimeICFusion.DEFAULT_STABLE_SIGN
